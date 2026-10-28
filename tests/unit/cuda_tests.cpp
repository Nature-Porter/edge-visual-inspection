#include "inspection/contracts.hpp"
#include "inspection/cuda_post.hpp"
#include <iostream>
#include <random>
#include <chrono>
using namespace inspection;
int main(){try{
    CudaStream stream;std::mt19937 gen(42);std::uniform_real_distribution<float> random(-2,2);
    for(size_t n:{1u,255u,256u,257u,65536u,65539u})for(int special=0;special<3;++special){
        std::vector<float> x(n);for(auto& v:x)v=random(gen);x[0]=.5f;if(special==1)x[n/2]=NAN;if(special==2)x[n/2]=INFINITY;
        auto ref=cpu_postprocess(x.data(),n,.5f);DeviceBuffer<float> input(n),blocks((n+255)/256),score(1);DeviceBuffer<unsigned char> mask(n);DeviceBuffer<int> invalid(1);
        cuda_check(cudaMemcpy(input.value,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));CudaEvent begin,end;
        cuda_check(cudaEventRecord(begin.value,stream.value));launch_postprocess(input.value,n,.5f,true,mask.value,blocks.value,invalid.value,score.value,stream.value);cuda_check(cudaEventRecord(end.value,stream.value));cuda_check(cudaEventSynchronize(end.value));
        float got,ms;int bad;std::vector<uint8_t> got_mask(n);cuda_check(cudaMemcpy(&got,score.value,4,cudaMemcpyDeviceToHost));cuda_check(cudaMemcpy(&bad,invalid.value,4,cudaMemcpyDeviceToHost));cuda_check(cudaMemcpy(got_mask.data(),mask.value,n,cudaMemcpyDeviceToHost));cuda_check(cudaEventElapsedTime(&ms,begin.value,end.value));
        if(ref.valid!=(bad==0)||(ref.valid&&got!=ref.score)||got_mask!=ref.mask)throw std::runtime_error("CUDA differs from CPU");
        std::cout<<json({{"elements",n},{"nonfinite_case",special},{"gpu_ms",ms},{"correct",true}}).dump()<<'\n';
    }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 5;}}
