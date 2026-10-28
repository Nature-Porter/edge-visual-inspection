#include "inspection/cuda_post.hpp"
#include <cmath>
namespace inspection {
__global__ void reduce_blocks(const float* in,size_t n,float t,bool make_mask,unsigned char* mask,float* blocks,int* invalid){
    __shared__ float shared[256];unsigned int tid=threadIdx.x;size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+tid;float x=-INFINITY;
    if(i<n){float v=in[i];if(!isfinite(v))atomicExch(invalid,1);else x=v;if(make_mask)mask[i]=v>=t?255:0;}
    shared[tid]=x;__syncthreads();for(unsigned int s=128;s;s>>=1){if(tid<s)shared[tid]=fmaxf(shared[tid],shared[tid+s]);__syncthreads();}if(!tid)blocks[blockIdx.x]=shared[0];
}
__global__ void reduce_final(const float* blocks,size_t n,float* score){
    __shared__ float shared[256];unsigned int t=threadIdx.x;float x=-INFINITY;for(size_t i=t;i<n;i+=256)x=fmaxf(x,blocks[i]);shared[t]=x;__syncthreads();
    for(unsigned int s=128;s;s>>=1){if(t<s)shared[t]=fmaxf(shared[t],shared[t+s]);__syncthreads();}if(!t)*score=shared[0];
}
void launch_postprocess(const float* in,size_t n,float t,bool make_mask,unsigned char* mask,float* blocks,int* invalid,float* score,cudaStream_t stream){
    if(!n)throw std::invalid_argument("empty CUDA map");cuda_check(cudaMemsetAsync(invalid,0,sizeof(int),stream));size_t b=(n+255)/256;
    reduce_blocks<<<static_cast<unsigned>(b),256,0,stream>>>(in,n,t,make_mask,mask,blocks,invalid);cuda_check(cudaGetLastError());
    reduce_final<<<1,256,0,stream>>>(blocks,b,score);cuda_check(cudaGetLastError());
}
}
