#include "inspection/contracts.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc.hpp>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/evp.h>
#endif
namespace inspection {
json read_json(const std::filesystem::path& p){std::ifstream f(p);if(!f)throw ConfigError("cannot open "+p.string());json j;f>>j;return j;}
void write_json(const std::filesystem::path& p,const json& j){std::ofstream f(p);f<<j.dump(2);f.flush();if(!f)throw std::runtime_error("write failed: "+p.string());}
std::string file_sha256(const std::filesystem::path& p){
    std::ifstream f(p,std::ios::binary);if(!f)throw ConfigError("missing hash input: "+p.string());
    unsigned char digest[32];char buf[65536];
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("BCrypt open failed");
    if(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0)<0){BCryptCloseAlgorithmProvider(alg,0);throw std::runtime_error("BCrypt create failed");}
    bool ok=true;while(f.read(buf,sizeof(buf))||f.gcount())if(BCryptHashData(hash,reinterpret_cast<PUCHAR>(buf),static_cast<ULONG>(f.gcount()),0)<0)ok=false;
    if(BCryptFinishHash(hash,digest,32,0)<0)ok=false;
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);if(!ok)throw std::runtime_error("BCrypt hash failed");
#else
    auto* ctx=EVP_MD_CTX_new();EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr);while(f.read(buf,sizeof(buf))||f.gcount())EVP_DigestUpdate(ctx,buf,static_cast<size_t>(f.gcount()));EVP_DigestFinal_ex(ctx,digest,nullptr);EVP_MD_CTX_free(ctx);
#endif
    std::ostringstream out;for(auto b:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(b);return out.str();
}
PostResult cpu_postprocess(const float* data,size_t n,std::optional<float> threshold){
    PostResult r;if(!n){r.valid=false;return r;}r.score=-std::numeric_limits<float>::infinity();if(threshold)r.mask.resize(n);
    for(size_t i=0;i<n;++i){if(!std::isfinite(data[i]))r.valid=false;else r.score=std::max(r.score,data[i]);if(threshold)r.mask[i]=data[i]>=*threshold?255:0;}return r;
}
std::vector<float> preprocess(const cv::Mat& image){
    if(image.empty()||image.type()!=CV_8UC3)throw std::runtime_error("INVALID_IMAGE");cv::Mat resized;cv::resize(image,resized,{256,256},0,0,cv::INTER_LINEAR);
    std::vector<float> out(3*256*256);for(int y=0;y<256;++y)for(int x=0;x<256;++x){auto bgr=resized.at<cv::Vec3b>(y,x);for(int c=0;c<3;++c)out[c*65536+y*256+x]=bgr[2-c]/255.0f;}return out;
}
Manifest::Manifest(const std::filesystem::path& path):doc(read_json(path)){
    if(doc.at("schema_version")!=1||doc.at("category")!="metal_nut")throw ConfigError("unsupported manifest version/category");
    for(auto key:{"model_id","checkpoint_sha256","tools","training_manifest_sha256","stats_manifest_sha256","requested_precision","validation_manifest_sha256"})if(!doc.contains(key))throw ConfigError(std::string("missing manifest ")+key);
    if(doc.at("score_reduction")!="max"||doc.at("preprocess_owner")!="external"||doc.at("normalization_owner")!="model")throw ConfigError("unsupported preprocessing ownership");
    if(doc.at("input").at("dtype")!="float32"||doc.at("output").at("dtype")!="float32"||doc.at("input").at("color")!="RGB"||doc.at("input").at("layout")!="NCHW"||doc.at("input").at("range")!=json::array({0,1})||doc.at("resize")!="opencv_inter_linear")throw ConfigError("unsupported I/O contract");
    input_shape=doc.at("input").at("shape").get<std::vector<int64_t>>();output_shape=doc.at("output").at("shape").get<std::vector<int64_t>>();
    if(input_shape!=std::vector<int64_t>{1,3,256,256}||output_shape.size()!=4||output_shape[0]!=1||output_shape[1]!=1||output_shape[2]<=0||output_shape[3]<=0)throw ConfigError("invalid shape");
    input_name=doc.at("input").at("name");output_name=doc.at("output").at("name");model_path=path.parent_path()/doc.at("onnx_path").get<std::string>();
    if(file_sha256(model_path)!=doc.at("onnx_sha256"))throw ConfigError("model SHA256 mismatch");
    if(!doc.at("image_threshold").is_null())image_threshold=doc.at("image_threshold").get<float>();
    if(!doc.at("pixel_threshold").is_null())pixel_threshold=doc.at("pixel_threshold").get<float>();
    if((image_threshold&&!std::isfinite(*image_threshold))||(pixel_threshold&&!std::isfinite(*pixel_threshold)))throw ConfigError("nonfinite threshold");
    if(doc.at("status")!="calibrated"&&doc.at("status")!="uncalibrated")throw ConfigError("unknown calibration status");
    if(doc.at("status")=="calibrated"&&(!image_threshold||doc.at("validation_manifest_sha256").is_null()))throw ConfigError("calibrated manifest without threshold provenance");
}
DirectoryReplaySource::DirectoryReplaySource(const std::filesystem::path& path){
    if(std::filesystem::is_directory(path)){
        for(const auto& entry:std::filesystem::directory_iterator(path))if(entry.is_regular_file()){auto ext=entry.path().extension().string();if(ext==".png"||ext==".jpg"||ext==".bmp")records_.push_back({{"path",std::filesystem::absolute(entry.path()).string()},{"source_image_id",file_sha256(entry.path())}});}
        std::sort(records_.begin(),records_.end(),[](auto&a,auto&b){return a["path"]<b["path"];});
    }else records_=read_json(path).at("records").get<std::vector<json>>();
    if(records_.empty())throw ConfigError("empty source");
}
}
