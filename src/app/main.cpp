#include "inspection/contracts.hpp"
#include "inspection/ort_backend.hpp"
#ifdef INSPECTION_TRT
#include "inspection/trt_backend.hpp"
#endif
#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace inspection;
using Clock=std::chrono::steady_clock;
struct PreciseWait {
#ifdef _WIN32
    HANDLE timer=CreateWaitableTimerExW(nullptr,nullptr,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_ALL_ACCESS);
    ~PreciseWait(){if(timer)CloseHandle(timer);}
    bool high_resolution() const{return timer!=nullptr;}
    void until(Clock::time_point target) const {
        auto remaining=target-Clock::now();
        if(remaining<=Clock::duration::zero())return;
        if(timer){
            using HundredNs=std::chrono::duration<int64_t,std::ratio<1,10000000>>;
            LARGE_INTEGER due{};due.QuadPart=-std::max<int64_t>(1,std::chrono::duration_cast<HundredNs>(remaining).count());
            if(SetWaitableTimerEx(timer,&due,0,nullptr,nullptr,nullptr,0)&&WaitForSingleObject(timer,INFINITE)==WAIT_OBJECT_0)return;
        }
        std::this_thread::sleep_until(target);
    }
#else
    bool high_resolution() const{return false;}
    void until(Clock::time_point target) const {std::this_thread::sleep_until(target);}
#endif
};
static int64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static std::string utc(){auto t=std::time(nullptr);std::tm tm{};
#ifdef _WIN32
gmtime_s(&tm,&t);
#else
gmtime_r(&t,&tm);
#endif
char buffer[32];std::strftime(buffer,sizeof(buffer),"%Y-%m-%dT%H:%M:%SZ",&tm);return buffer;}
static json stats(std::vector<double> v){if(v.empty())return nullptr;std::sort(v.begin(),v.end());auto q=[&](double p){return v[static_cast<size_t>(std::ceil(p*v.size()))-1];};return {{"count",v.size()},{"p50",q(.5)},{"p95",q(.95)},{"p99",q(.99)}};}
struct Packet {json event;cv::Mat map;std::string source;int64_t decision_ns=0;};
int main(int argc,char** argv){
 try{
    if(argc<2){std::cerr<<"inspection run|benchmark --manifest PATH --source PATH --backend ort-cuda|ort-cpu|trt-fp32|trt-int8 --output DIR\n";return configuration;}
    std::string command=argv[1];if(command!="run"&&command!="benchmark")throw ConfigError("unknown command");
    std::map<std::string,std::string> args;
    const std::vector<std::string> allowed={"--manifest","--source","--backend","--engine","--engine-evidence","--output","--fps","--deadline-ms","--queue-capacity","--postprocess","--mode","--warmup","--iterations","--duration-seconds","--inject-delay-ms","--inject-inference-error-every","--inject-write-error-after","--loop","--profile"};
    for(int i=2;i<argc;++i){std::string key=argv[i];if(std::find(allowed.begin(),allowed.end(),key)==allowed.end())throw ConfigError("unknown argument "+key);if(key=="--loop"||key=="--profile")args[key]="true";else{if(++i>=argc)throw ConfigError("missing argument value");args[key]=argv[i];}}
    auto get=[&](const std::string& k,const std::string& d){auto it=args.find(k);return it==args.end()?d:it->second;};
    for(auto key:{"--manifest","--source","--output"})if(!args.count(key))throw ConfigError(std::string("required ")+key);
    auto parse=[&](const std::string& k,double d){std::string s=get(k,std::to_string(d));size_t n;double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v)||v<0)throw ConfigError("invalid "+k);return v;};
    double fps=parse("--fps",20),deadline=parse("--deadline-ms",50),duration=parse("--duration-seconds",0),delay=parse("--inject-delay-ms",0);
    size_t capacity=static_cast<size_t>(parse("--queue-capacity",4));if(!fps||!deadline||!capacity)throw ConfigError("fps/deadline/capacity must be positive");
    bool loop=args.count("--loop"),profile=args.count("--profile"),benchmark=command=="benchmark";
    if(loop&&!duration)throw ConfigError("--loop requires finite --duration-seconds");
    std::string mode=get("--mode","pipeline"),backend_name=get("--backend","ort-cuda"),post=get("--postprocess","cpu");
    if(post!="cpu"&&post!="cuda")throw ConfigError("invalid postprocess");if(mode!="model"&&mode!="pipeline"&&mode!="replay")throw ConfigError("invalid benchmark mode");
    auto manifest=Manifest(args.at("--manifest"));DirectoryReplaySource source(args.at("--source"));
    std::filesystem::path out=args.at("--output");std::filesystem::create_directories(out);
    auto session_id=std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    if(std::filesystem::exists(out/"events.jsonl"))throw ConfigError("output already contains a run; choose a new directory");
    std::unique_ptr<IInferenceBackend> backend;
    std::optional<std::string> engine_sha,evidence_sha;
    if(backend_name=="trt-int8"||backend_name=="trt-fp32"){
        if(profile)throw ConfigError("--profile is ORT-only; TensorRT layer profile is recorded in engine evidence");
        if(!args.count("--engine")||!args.count("--engine-evidence"))throw ConfigError("TensorRT backend requires --engine and --engine-evidence");
#ifdef INSPECTION_TRT
        auto trt=std::make_unique<TrtBackend>(manifest,args.at("--engine"),args.at("--engine-evidence"),backend_name=="trt-int8");
        engine_sha=trt->engine_sha256();evidence_sha=trt->evidence_sha256();backend=std::move(trt);
#else
        throw BackendError("built without TensorRT backend");
#endif
    }else{
        if(args.count("--engine")||args.count("--engine-evidence"))throw ConfigError("engine arguments require TensorRT backend");
        backend=std::make_unique<OrtBackend>(manifest,backend_name,profile?out/"ort-profile":std::filesystem::path{});
    }
    PreciseWait schedule_wait;
    write_json(out/"snapshot.json",{{"schema_version",1},{"session_id",session_id},{"manifest",manifest.doc},{"manifest_sha256",file_sha256(args.at("--manifest"))},{"args",args},{"fps",fps},{"deadline_ms",deadline},{"queue_capacity",capacity},{"source",source.records()},{"ort_version",OrtGetApiBase()->GetVersionString()},{"engine_sha256",engine_sha?json(*engine_sha):json(nullptr)},{"engine_evidence_sha256",evidence_sha?json(*evidence_sha):json(nullptr)},{"scheduler_wait",schedule_wait.high_resolution()?"windows_high_resolution_waitable_timer":"standard_sleep_until"},{"timing","decision from scheduled steady-clock; artifact storage excluded"}});
    size_t warmup=static_cast<size_t>(parse("--warmup",100)),iterations=static_cast<size_t>(parse("--iterations",1000));
    if(benchmark&&mode!="replay"){
        if(!iterations)throw ConfigError("iterations must be positive");
        cv::Mat image=cv::imread(source.records()[0].at("path").get<std::string>());auto input=preprocess(image);
        for(size_t i=0;i<warmup;++i)backend->run(input,post=="cuda",false);
        std::vector<double> wall,gpu,posttimes;std::ofstream raw(out/"benchmark.jsonl");if(!raw)throw std::runtime_error("benchmark output open failed");
        for(size_t i=0;i<iterations;++i){auto start=Clock::now();if(mode=="pipeline")input=preprocess(image);auto r=backend->run(input,post=="cuda",false);auto status=r.post.valid?"SUCCESS":"INVALID_OUTPUT";auto dec=decision(status,r.post.valid?std::optional<float>(r.post.score):std::nullopt,manifest.image_threshold);double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();wall.push_back(ms);posttimes.push_back(r.postprocess_ms);if(r.model_gpu_ms)gpu.push_back(*r.model_gpu_ms);raw<<json({{"iteration",i},{"wall_ms",ms},{"model_gpu_ms",r.model_gpu_ms?json(*r.model_gpu_ms):json(nullptr)},{"status",status},{"decision",dec}}).dump()<<'\n';}
        raw.flush();if(!raw)throw std::runtime_error("benchmark write failed");write_json(out/"summary.json",{{"mode",mode},{"backend",backend_name},{"wall_ms",stats(wall)},{"model_gpu_ms",stats(gpu)},{"postprocess_ms",stats(posttimes)},{"model_sha256",manifest.doc["onnx_sha256"]},{"quality_claim",false},{"exit_code",0}});
        if(profile)std::cout<<backend->finish_profile()<<'\n';return success;
    }
    // Warm-up completes before the scheduled arrival clock starts. Skip undecodable sources.
    for(const auto& record:source.records()){
        auto image=cv::imread(record.at("path").get<std::string>(),cv::IMREAD_COLOR);
        if(!image.empty()){auto tensor=preprocess(image);for(size_t i=0;i<warmup;++i)backend->run(tensor,post=="cuda",false);break;}
    }
    BoundedQueue<Frame> queue(capacity);BoundedQueue<Packet> events(64);std::atomic<bool> failed{false};std::atomic<uint64_t> generated{0};std::string failure;
    uint64_t completed=0,successful=0,reviews=0,errors=0,overflows=0,timeouts=0;std::vector<double> latency,pipeline,gpu,artifact;
    const auto inject_error=static_cast<uint64_t>(parse("--inject-inference-error-every",0));const auto inject_write=static_cast<uint64_t>(parse("--inject-write-error-after",0));
    auto make_event=[&](const Frame& f,const std::string& status){return json{{"schema_version",1},{"session_id",session_id},{"frame_id",f.id},{"source_image_id",f.source.at("source_image_id")},{"scheduled_monotonic_ns",f.scheduled_ns},{"arrival_time_utc",f.utc},{"status",status},{"decision","REVIEW"},{"score",nullptr},{"image_threshold",manifest.doc["image_threshold"]},{"pixel_threshold",manifest.doc["pixel_threshold"]},{"decision_latency_ms",(now_ns()-f.scheduled_ns)/1e6},{"pipeline_ms",nullptr},{"model_gpu_ms",nullptr},{"queue_ms",nullptr},{"decode_ms",nullptr},{"postprocess_ms",nullptr},{"model_sha256",manifest.doc["onnx_sha256"]},{"requested_backend",backend_name},{"observed_provider_summary",backend_name=="trt-int8"||backend_name=="trt-fp32"?"TensorRT engine direct CUDA runtime; no ORT EP":profile?"pending_profile":"not_profiled"},{"declared_precision",manifest.doc["requested_precision"]},{"verified_precision_evidence_id",evidence_sha?json(*evidence_sha):json(nullptr)},{"anomaly_map_path",nullptr},{"artifact_error",nullptr},{"artifact_latency_ms",nullptr}};};
    const int64_t start=now_ns();
    std::thread writer([&]{try{
        std::ofstream log(out/"events.jsonl");if(!log)throw std::runtime_error("event open failed");Packet packet;
        while(events.pop(packet)){
            auto& e=packet.event;
            if(inject_write&&completed>=inject_write)throw std::runtime_error("injected JSONL write failure");
            if(!packet.map.empty()&&e["decision"]=="REVIEW"){
                try{auto name="map-"+std::to_string(e["frame_id"].get<uint64_t>())+".tiff";if(!cv::imwrite((out/name).string(),packet.map))throw std::runtime_error("imwrite returned false");e["anomaly_map_path"]=name;
                    cv::Mat preview,colour;cv::normalize(packet.map,preview,0,255,cv::NORM_MINMAX,CV_8U);cv::applyColorMap(preview,colour,cv::COLORMAP_JET);if(!cv::imwrite((out/("heat-"+std::to_string(e["frame_id"].get<uint64_t>())+".png")).string(),colour))throw std::runtime_error("heatmap write failed");
                }catch(const std::exception& ex){e["artifact_error"]=ex.what();}
                e["artifact_latency_ms"]=(now_ns()-packet.decision_ns)/1e6;artifact.push_back(e["artifact_latency_ms"]);
            }
            log<<e.dump()<<'\n';log.flush();if(!log)throw std::runtime_error("JSONL write failed");
            ++completed;std::string status=e["status"];successful+=!e["score"].is_null();reviews+=e["decision"]=="REVIEW";overflows+=status=="QUEUE_OVERFLOW";timeouts+=status=="TIMEOUT";errors+=status=="INVALID_IMAGE"||status=="INVALID_OUTPUT"||status=="INFERENCE_ERROR";
            latency.push_back(e["decision_latency_ms"]);if(!e["pipeline_ms"].is_null())pipeline.push_back(e["pipeline_ms"]);if(!e["model_gpu_ms"].is_null())gpu.push_back(e["model_gpu_ms"]);
        }
    }catch(const std::exception& e){failure=e.what();failed=true;queue.close();events.close();}});
    std::thread worker([&]{Frame f;while(queue.pop(f)){if(failed)break;Packet p;p.event=make_event(f,"SUCCESS");auto& e=p.event;
        auto begin=now_ns();e["queue_ms"]=(begin-f.scheduled_ns)/1e6;
        if((begin-f.scheduled_ns)/1e6>=deadline)e["status"]="TIMEOUT";
        else try{
            auto image=cv::imread(f.source.at("path").get<std::string>(),cv::IMREAD_COLOR);auto decoded=now_ns();e["decode_ms"]=(decoded-begin)/1e6;
            if(image.empty())e["status"]="INVALID_IMAGE";
            else{
                if(delay)std::this_thread::sleep_for(std::chrono::duration<double,std::milli>(delay));
                if(inject_error&&f.id%inject_error==0)throw std::runtime_error("injected inference error");
                auto r=backend->run(preprocess(image),post=="cuda",true);p.map=r.map;
                e["pipeline_ms"]=(now_ns()-decoded)/1e6;e["postprocess_ms"]=r.postprocess_ms;if(r.model_gpu_ms)e["model_gpu_ms"]=*r.model_gpu_ms;
                if(!r.post.valid)e["status"]="INVALID_OUTPUT";else{e["score"]=r.post.score;if(manifest.doc["status"]!="calibrated")e["status"]="UNCALIBRATED";}
            }
        }catch(const std::exception& ex){e["status"]="INFERENCE_ERROR";e["error"]=ex.what();}
        if(e["status"]=="SUCCESS"||e["status"]=="UNCALIBRATED")if((now_ns()-f.scheduled_ns)/1e6>=deadline)e["status"]="TIMEOUT";
        std::optional<float> score;if(!e["score"].is_null())score=e["score"].get<float>();e["decision"]=decision(e["status"],score,manifest.image_threshold);p.decision_ns=now_ns();e["decision_latency_ms"]=(p.decision_ns-f.scheduled_ns)/1e6;
        if(!events.push(std::move(p)))break;
    }});
    uint64_t count=0;auto& records=source.records();
    while(!failed){int64_t scheduled=start+static_cast<int64_t>(count*1e9/fps);if(duration&&scheduled-start>duration*1e9)break;if(!loop&&count>=records.size())break;
        schedule_wait.until(Clock::time_point(std::chrono::nanoseconds(scheduled)));if(failed)break;
        Frame f{++count,records[(count-1)%records.size()],scheduled,utc()};generated=count;
        if(!queue.try_push(f)){Packet p;p.event=make_event(f,"QUEUE_OVERFLOW");p.decision_ns=now_ns();if(!events.push(std::move(p)))break;}
    }
    queue.close();worker.join();events.close();writer.join();double elapsed=(now_ns()-start)/1e9;
    json summary={{"generated",generated.load()},{"completed",completed},{"successful_inference",successful},{"review",reviews},{"error",errors},{"queue_overflow",overflows},{"timeout",timeouts},{"completed_fps",completed/elapsed},{"elapsed_seconds",elapsed},{"decision_latency_ms",stats(latency)},{"pipeline_ms",stats(pipeline)},{"model_gpu_ms",stats(gpu)},{"artifact_latency_ms",stats(artifact)},{"process_gpu_memory_bytes",nullptr},{"gpu_memory_method","WDDM process memory unavailable; external sampling required for a memory trend"},{"exit_code",failed?4:0},{"failure",failed?json(failure):json(nullptr)},{"frame_coverage",generated?static_cast<double>(completed)/generated.load():0},{"latency_pass",!latency.empty()&&stats(latency)["p95"].get<double>()<=40&&static_cast<double>(timeouts)/std::max<uint64_t>(1,generated)<=.01}};
    summary["fixture"]=manifest.doc.value("fixture",false);
    summary["latency_pass"]=!failed&&!manifest.doc.value("fixture",false)&&manifest.doc["status"]=="calibrated"&&completed==generated.load()&&overflows==0&&errors==0&&summary["latency_pass"].get<bool>();
    write_json(out/"summary.json",summary);if(profile)std::cout<<backend->finish_profile()<<'\n';return failed?infrastructure:success;
 }catch(const ConfigError& e){std::cerr<<e.what()<<'\n';return configuration;}catch(const BackendError& e){std::cerr<<e.what()<<'\n';return dependency;}catch(const json::exception& e){std::cerr<<e.what()<<'\n';return configuration;}catch(const std::invalid_argument& e){std::cerr<<e.what()<<'\n';return configuration;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return infrastructure;}
}
