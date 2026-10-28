#pragma once
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
namespace inspection {
using json = nlohmann::json;
enum ExitCode { success=0, configuration=2, dependency=3, infrastructure=4, acceptance=5 };
struct ConfigError:std::runtime_error { using runtime_error::runtime_error; };
struct BackendError:std::runtime_error { using runtime_error::runtime_error; };
struct PostResult { float score=0; bool valid=true; std::vector<uint8_t> mask; };
PostResult cpu_postprocess(const float* data, size_t count, std::optional<float> threshold);
std::vector<float> preprocess(const cv::Mat& image);
std::string file_sha256(const std::filesystem::path& path);
json read_json(const std::filesystem::path& path);
void write_json(const std::filesystem::path& path,const json& value);
struct Manifest {
    json doc;
    std::filesystem::path model_path;
    std::vector<int64_t> input_shape, output_shape;
    std::string input_name,output_name;
    std::optional<float> image_threshold,pixel_threshold;
    explicit Manifest(const std::filesystem::path& path);
};
inline std::string decision(const std::string& status,std::optional<float> score,std::optional<float> threshold) {
    return status=="SUCCESS" && score && std::isfinite(*score) && threshold && *score<*threshold ? "OK":"REVIEW";
}
template<class T> class BoundedQueue {
    std::mutex mutex_; std::condition_variable cv_; std::deque<T> queue_; size_t capacity_; bool closed_=false;
public:
    explicit BoundedQueue(size_t n):capacity_(n) {if(!n) throw ConfigError("queue capacity must be positive");}
    bool try_push(T value) {std::lock_guard<std::mutex> lock(mutex_); if(closed_||queue_.size()>=capacity_)return false; queue_.push_back(std::move(value));cv_.notify_one();return true;}
    bool push(T value) {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return closed_||queue_.size()<capacity_;});if(closed_)return false;queue_.push_back(std::move(value));cv_.notify_all();return true;}
    bool pop(T& value) {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return closed_||!queue_.empty();});if(queue_.empty())return false;value=std::move(queue_.front());queue_.pop_front();cv_.notify_all();return true;}
    void close() {std::lock_guard<std::mutex> lock(mutex_);closed_=true;cv_.notify_all();}
};
struct Frame {uint64_t id;json source;int64_t scheduled_ns;std::string utc;};
struct InferenceResult {PostResult post;cv::Mat map;std::optional<double> model_gpu_ms;double postprocess_ms=0;};
class IInferenceBackend {
public:
    virtual ~IInferenceBackend()=default;
    virtual InferenceResult run(const std::vector<float>& input,bool cuda_post,bool keep_map)=0;
    virtual std::string finish_profile(){return {};}
};
class IFrameSource {public:virtual ~IFrameSource()=default;virtual const std::vector<json>& records()const=0;};
class DirectoryReplaySource final:public IFrameSource {
    std::vector<json> records_;
public:
    explicit DirectoryReplaySource(const std::filesystem::path& path);
    const std::vector<json>& records()const override{return records_;}
};
}
