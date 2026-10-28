#include "inspection/trt_backend.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

namespace inspection {
void TrtBackend::Logger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
}

static bool same_shape(nvinfer1::Dims dims, const std::vector<int64_t>& expected) {
    if (dims.nbDims != static_cast<int>(expected.size())) return false;
    for (int i = 0; i < dims.nbDims; ++i) if (dims.d[i] != expected[static_cast<size_t>(i)]) return false;
    return true;
}

TrtBackend::TrtBackend(const Manifest& manifest, const std::filesystem::path& engine_path,
                       const std::filesystem::path& evidence_path, bool expect_int8)
    : manifest_(manifest), input_(3 * 256 * 256),
      output_(static_cast<size_t>(manifest.output_shape[2] * manifest.output_shape[3])),
      blocks_((output_.count + 255) / 256), score_(1), mask_(output_.count), invalid_(1) {
    // Only a locally built, pinned engine with real layer/runtime evidence can claim INT8.
    const auto evidence = read_json(evidence_path);
    evidence_sha256_ = file_sha256(evidence_path);
    engine_sha256_ = file_sha256(engine_path);
    const bool identity_ok = evidence.value("model_sha256", "") == manifest.doc.at("onnx_sha256") &&
                             evidence.value("engine_sha256", "") == engine_sha256_;
    const bool precision_ok = expect_int8 ?
        (evidence.value("status", "") == "passed_validation_experiment" &&
         evidence.value("int8_execution_verified", false) &&
         evidence.value("validation_quality_pass", false) &&
         evidence.value("int8_conv_layer_count", 0) >= 1 &&
         manifest.doc.at("requested_precision") == "INT8/FP32") :
        (evidence.value("status", "") == "built" &&
         evidence.value("exit_code", -1) == 0 &&
         evidence.value("int8_compute_count", -1) == 0 &&
         manifest.doc.at("requested_precision") == "FP32");
    if (!identity_ok || !precision_ok)
        throw ConfigError("TensorRT engine/model/evidence mismatch or precision unverified");
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw ConfigError("cannot open TensorRT engine");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.empty()) throw ConfigError("empty TensorRT engine");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw BackendError("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw BackendError("TensorRT engine deserialization failed");
    if (engine_->getNbIOTensors() != 2) throw ConfigError("TensorRT engine must have one input and output");
    auto check_tensor = [&](const std::string& name, nvinfer1::TensorIOMode mode,
                            const std::vector<int64_t>& shape) {
        if (engine_->getTensorIOMode(name.c_str()) != mode ||
            engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorLocation(name.c_str()) != nvinfer1::TensorLocation::kDEVICE ||
            !same_shape(engine_->getTensorShape(name.c_str()), shape))
            throw ConfigError("TensorRT tensor name/type/shape/location mismatch: " + name);
    };
    check_tensor(manifest.input_name, nvinfer1::TensorIOMode::kINPUT, manifest.input_shape);
    check_tensor(manifest.output_name, nvinfer1::TensorIOMode::kOUTPUT, manifest.output_shape);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw BackendError("TensorRT execution context creation failed");
    if (!context_->setTensorAddress(manifest.input_name.c_str(), input_.value) ||
        !context_->setTensorAddress(manifest.output_name.c_str(), output_.value))
        throw BackendError("TensorRT tensor binding failed");
}

TrtBackend::~TrtBackend() {
    cudaStreamSynchronize(stream_.value);
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

InferenceResult TrtBackend::run(const std::vector<float>& input, bool cuda_post, bool keep_map) {
    if (input.size() != input_.count) throw ConfigError("TensorRT input size mismatch");
    const auto n = output_.count;
    auto stream = stream_.value;
    struct SyncGuard { cudaStream_t stream; ~SyncGuard() { cudaStreamSynchronize(stream); } } sync{stream};
    InferenceResult result;
    std::vector<float> host(n);
    CudaEvent begin, end, post_begin, post_end;
    cuda_check(cudaMemcpyAsync(input_.value, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaEventRecord(begin.value, stream));
    if (!context_->enqueueV3(stream)) throw BackendError("TensorRT enqueueV3 failed");
    cuda_check(cudaEventRecord(end.value, stream));
    if (cuda_post) {
        float score = 0;
        int invalid = 0;
        cuda_check(cudaEventRecord(post_begin.value, stream));
        launch_postprocess(output_.value, n, manifest_.pixel_threshold.value_or(0),
                           manifest_.pixel_threshold.has_value(), mask_.value, blocks_.value,
                           invalid_.value, score_.value, stream);
        cuda_check(cudaEventRecord(post_end.value, stream));
        cuda_check(cudaMemcpyAsync(&score, score_.value, sizeof(score), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaMemcpyAsync(&invalid, invalid_.value, sizeof(invalid), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        result.post.score = score;
        result.post.valid = invalid == 0 && std::isfinite(score);
        keep_map = keep_map && (!result.post.valid || manifest_.doc["status"] != "calibrated" ||
                                !manifest_.image_threshold || score >= *manifest_.image_threshold);
        if (keep_map) {
            cuda_check(cudaMemcpyAsync(host.data(), output_.value, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
        }
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, post_begin.value, post_end.value));
        result.postprocess_ms = ms;
    } else {
        cuda_check(cudaMemcpyAsync(host.data(), output_.value, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        const auto start = std::chrono::steady_clock::now();
        result.post = cpu_postprocess(host.data(), n, manifest_.pixel_threshold);
        result.postprocess_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    float model_ms = 0;
    cuda_check(cudaEventElapsedTime(&model_ms, begin.value, end.value));
    result.model_gpu_ms = model_ms;
    if (keep_map) result.map = cv::Mat(static_cast<int>(manifest_.output_shape[2]),
                                       static_cast<int>(manifest_.output_shape[3]), CV_32F, host.data()).clone();
    return result;
}
}
