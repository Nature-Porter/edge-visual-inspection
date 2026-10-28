#include <cuda_runtime.h>
#include <cstdio>
__global__ void add_one(int* x){*x+=1;}
int main(){int* device=nullptr;int value=41;if(cudaMalloc(&device,sizeof(int))!=cudaSuccess)return 3;
if(cudaMemcpy(device,&value,sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(device);return 5;}
add_one<<<1,1>>>(device);auto error=cudaDeviceSynchronize();if(error==cudaSuccess)error=cudaMemcpy(&value,device,sizeof(int),cudaMemcpyDeviceToHost);cudaFree(device);
std::printf("CUDA add_one result=%d\n",value);return error==cudaSuccess&&value==42?0:5;}
