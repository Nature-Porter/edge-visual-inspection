#include "inspection/contracts.hpp"
#include <iostream>
#include <thread>
#include <limits>
using namespace inspection;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(0)
int main(){try{
    CHECK(success==0&&configuration==2&&dependency==3&&infrastructure==4&&acceptance==5);
    float a[]={-3,0,.5f,.5f};auto p=cpu_postprocess(a,4,.5f);CHECK(p.valid&&p.score==.5f);CHECK(p.mask==std::vector<uint8_t>({0,0,255,255}));
    CHECK(decision("SUCCESS",.5f,.5f)=="REVIEW");CHECK(decision("SUCCESS",.49f,.5f)=="OK");
    for(auto s:{"INVALID_IMAGE","INVALID_OUTPUT","INFERENCE_ERROR","TIMEOUT","QUEUE_OVERFLOW","UNCALIBRATED"})CHECK(decision(s,0.f,1.f)=="REVIEW");
    a[1]=std::numeric_limits<float>::quiet_NaN();CHECK(!cpu_postprocess(a,4,.5f).valid);a[1]=std::numeric_limits<float>::infinity();CHECK(!cpu_postprocess(a,4,{}).valid);
    BoundedQueue<int> q(1);CHECK(q.try_push(1));CHECK(!q.try_push(2));int v;CHECK(q.pop(v)&&v==1);q.close();CHECK(!q.pop(v));CHECK(!q.try_push(3));
    BoundedQueue<int> concurrent(4);std::thread producer([&]{for(int i=0;i<10000;++i)CHECK(concurrent.push(i));concurrent.close();});int count=0;while(concurrent.pop(v)){CHECK(v==count);++count;}producer.join();CHECK(count==10000);
    cv::Mat image(256,256,CV_8UC3,cv::Scalar(0,127,255));auto x=preprocess(image);CHECK(x[0]==1.f&&x[65536]==127/255.f&&x[131072]==0.f);
    std::cout<<"core correctness passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 5;}}
