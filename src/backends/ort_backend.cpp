#include "inspection/ort_backend.hpp"
#include <chrono>
#include <numeric>
namespace inspection {
OrtBackend::OrtBackend(const Manifest& m,const std::string& backend,const std::filesystem::path& profile):manifest_(m),gpu_(backend=="ort-cuda"){
    if(backend!="ort-cuda"&&backend!="ort-cpu")throw BackendError("backend unavailable; TRT requires separately verified engine evidence");
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if(!profile.empty())options_.EnableProfiling(profile.c_str());
    if(gpu_){
#ifdef INSPECTION_CUDA
        try{
            stream_=std::make_unique<CudaStream>();
            OrtCUDAProviderOptionsV2* cuda=nullptr;Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
            std::unique_ptr<OrtCUDAProviderOptionsV2,void(*)(OrtCUDAProviderOptionsV2*)> guard(cuda,Ort::GetApi().ReleaseCUDAProviderOptions);
            std::string address=std::to_string(reinterpret_cast<uintptr_t>(stream_->value));
            const char* keys[]={"device_id","user_compute_stream","do_copy_in_default_stream","use_tf32"};const char* vals[]={"0",address.c_str(),"1","0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda,keys,vals,4));options_.AppendExecutionProvider_CUDA_V2(*cuda);
            // Do not silently run the requested GPU model on CPU.
            options_.AddConfigEntry("session.disable_cpu_ep_fallback","1");
            size_t n=static_cast<size_t>(m.output_shape[2]*m.output_shape[3]);
            input_=std::make_unique<DeviceBuffer<float>>(3*256*256);output_=std::make_unique<DeviceBuffer<float>>(n);blocks_=std::make_unique<DeviceBuffer<float>>((n+255)/256);score_=std::make_unique<DeviceBuffer<float>>(1);invalid_=std::make_unique<DeviceBuffer<int>>(1);mask_=std::make_unique<DeviceBuffer<unsigned char>>(n);
        }catch(const std::exception& e){throw BackendError(e.what());}
#else
        throw BackendError("built without CUDA; use explicit ort-cpu only for software tests");
#endif
    }
    try{session_=std::make_unique<Ort::Session>(env_,m.model_path.c_str(),options_);}catch(const Ort::Exception& e){throw BackendError(e.what());}
    Ort::AllocatorWithDefaultOptions allocator;
    if(session_->GetInputCount()!=1||session_->GetOutputCount()!=1)throw ConfigError("expected exactly one input/output");
    if(m.input_name!=session_->GetInputNameAllocated(0,allocator).get()||m.output_name!=session_->GetOutputNameAllocated(0,allocator).get())throw ConfigError("ONNX names differ from manifest");
    auto input_type=session_->GetInputTypeInfo(0);auto output_type=session_->GetOutputTypeInfo(0);
    auto in=input_type.GetTensorTypeAndShapeInfo();auto out=output_type.GetTensorTypeAndShapeInfo();
    if(in.GetShape()!=m.input_shape||out.GetShape()!=m.output_shape||in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)throw ConfigError("ONNX tensor contract mismatch");
}
OrtBackend::~OrtBackend(){
#ifdef INSPECTION_CUDA
    if(stream_)cudaStreamSynchronize(stream_->value);
#endif
    session_.reset();
}
std::string OrtBackend::finish_profile(){Ort::AllocatorWithDefaultOptions a;return session_->EndProfilingAllocated(a).get();}
InferenceResult OrtBackend::run(const std::vector<float>& input,bool cuda_post,bool keep_map){
    if(input.size()!=3*256*256)throw ConfigError("input size mismatch");
    InferenceResult result;size_t n=static_cast<size_t>(manifest_.output_shape[2]*manifest_.output_shape[3]);std::vector<float> host(n);
    const char* input_names[]={manifest_.input_name.c_str()};const char* output_names[]={manifest_.output_name.c_str()};
    if(gpu_){
#ifdef INSPECTION_CUDA
        auto s=stream_->value;CudaEvent start,end,post_start,post_end;
        // Guard synchronizes before local host buffers or Ort::Value can be released on errors.
        struct SyncGuard {cudaStream_t s;~SyncGuard(){cudaStreamSynchronize(s);}} sync{s};
        cuda_check(cudaMemcpyAsync(input_->value,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,s));
        Ort::MemoryInfo memory("Cuda",OrtAllocatorType::OrtDeviceAllocator,0,OrtMemTypeDefault);
        auto in=Ort::Value::CreateTensor<float>(memory,input_->value,input.size(),manifest_.input_shape.data(),4);
        auto out=Ort::Value::CreateTensor<float>(memory,output_->value,n,manifest_.output_shape.data(),4);
        Ort::IoBinding binding(*session_);binding.BindInput(input_names[0],in);binding.BindOutput(output_names[0],out);
        cuda_check(cudaEventRecord(start.value,s));session_->Run(Ort::RunOptions{nullptr},binding);cuda_check(cudaEventRecord(end.value,s));
        if(cuda_post){
            int invalid=0;float score=0;cuda_check(cudaEventRecord(post_start.value,s));
            launch_postprocess(output_->value,n,manifest_.pixel_threshold.value_or(0),manifest_.pixel_threshold.has_value(),mask_->value,blocks_->value,invalid_->value,score_->value,s);
            cuda_check(cudaEventRecord(post_end.value,s));
            cuda_check(cudaMemcpyAsync(&score,score_->value,sizeof(float),cudaMemcpyDeviceToHost,s));cuda_check(cudaMemcpyAsync(&invalid,invalid_->value,sizeof(int),cudaMemcpyDeviceToHost,s));
            cuda_check(cudaStreamSynchronize(s));result.post.score=score;result.post.valid=invalid==0&&std::isfinite(score);
            keep_map=keep_map&&(!result.post.valid||manifest_.doc["status"]!="calibrated"||!manifest_.image_threshold||score>=*manifest_.image_threshold);
            if(keep_map){cuda_check(cudaMemcpyAsync(host.data(),output_->value,n*sizeof(float),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));}
            float ms;cuda_check(cudaEventElapsedTime(&ms,post_start.value,post_end.value));result.postprocess_ms=ms;
        }else{cuda_check(cudaMemcpyAsync(host.data(),output_->value,n*sizeof(float),cudaMemcpyDeviceToHost,s));cuda_check(cudaStreamSynchronize(s));}
        float ms;cuda_check(cudaEventElapsedTime(&ms,start.value,end.value));result.model_gpu_ms=ms;
#endif
    }else{
        if(cuda_post)throw BackendError("CUDA postprocessing requires GPU backend");
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto in=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(input.data()),input.size(),manifest_.input_shape.data(),4);
        auto out=session_->Run(Ort::RunOptions{nullptr},input_names,&in,1,output_names,1);
        std::copy_n(out[0].GetTensorData<float>(),n,host.data());
    }
    if(!cuda_post){auto start=std::chrono::steady_clock::now();result.post=cpu_postprocess(host.data(),n,manifest_.pixel_threshold);result.postprocess_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
    if(keep_map)result.map=cv::Mat(static_cast<int>(manifest_.output_shape[2]),static_cast<int>(manifest_.output_shape[3]),CV_32F,host.data()).clone();
    return result;
}
}
