#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstddef>
namespace inspection {
inline void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e));}
struct CudaStream {cudaStream_t value{};CudaStream(){cuda_check(cudaStreamCreate(&value));}~CudaStream(){cudaStreamSynchronize(value);cudaStreamDestroy(value);}CudaStream(const CudaStream&)=delete;};
struct CudaEvent {cudaEvent_t value{};CudaEvent(){cuda_check(cudaEventCreate(&value));}~CudaEvent(){cudaEventDestroy(value);}CudaEvent(const CudaEvent&)=delete;};
template<class T> struct DeviceBuffer {T* value=nullptr;size_t count;explicit DeviceBuffer(size_t n):count(n){cuda_check(cudaMalloc(reinterpret_cast<void**>(&value),n*sizeof(T)));}~DeviceBuffer(){cudaFree(value);}DeviceBuffer(const DeviceBuffer&)=delete;};
// Caller owns all storage until stream completion; invalid includes any NaN or infinity.
void launch_postprocess(const float* input,size_t n,float threshold,bool make_mask,unsigned char* mask,float* blocks,int* invalid,float* score,cudaStream_t stream);
}
