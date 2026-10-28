#pragma once
#include "inspection/contracts.hpp"
#include "inspection/cuda_post.hpp"
#include <NvInfer.h>
#include <memory>

namespace inspection {
class TrtBackend final : public IInferenceBackend {
    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    } logger_;
    Manifest manifest_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    CudaStream stream_;
    DeviceBuffer<float> input_;
    DeviceBuffer<float> output_;
    DeviceBuffer<float> blocks_;
    DeviceBuffer<float> score_;
    DeviceBuffer<unsigned char> mask_;
    DeviceBuffer<int> invalid_;
    std::string engine_sha256_, evidence_sha256_;
public:
    TrtBackend(const Manifest& manifest, const std::filesystem::path& engine_path,
               const std::filesystem::path& evidence_path, bool expect_int8);
    ~TrtBackend() override;
    InferenceResult run(const std::vector<float>& input, bool cuda_post, bool keep_map) override;
    const std::string& engine_sha256() const { return engine_sha256_; }
    const std::string& evidence_sha256() const { return evidence_sha256_; }
};
}
