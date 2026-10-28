#pragma once
#include "inspection/contracts.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#ifdef INSPECTION_CUDA
#include "inspection/cuda_post.hpp"
#endif
namespace inspection {
class OrtBackend final:public IInferenceBackend {
    Manifest manifest_;bool gpu_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"inspection"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
#ifdef INSPECTION_CUDA
    std::unique_ptr<CudaStream> stream_;
    std::unique_ptr<DeviceBuffer<float>> input_,output_,blocks_,score_;
    std::unique_ptr<DeviceBuffer<unsigned char>> mask_;
    std::unique_ptr<DeviceBuffer<int>> invalid_;
#endif
public:
    OrtBackend(const Manifest& manifest,const std::string& backend,const std::filesystem::path& profile={});
    ~OrtBackend() override;
    InferenceResult run(const std::vector<float>& input,bool cuda_post,bool keep_map) override;
    std::string finish_profile() override;
};
}
