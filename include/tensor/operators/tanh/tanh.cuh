
#ifndef tanh_CUH
#define tanh_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor/tensor.h"

__global__ void tanh_kernel(float *a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        auto aa = a[idx];
        auto ax = exp(aa);
        auto bx = exp(-aa);
        a[idx] = (ax-bx)/(ax+bx);
    }
}


void tanh_cuda_kernel(void* input, size_t size, TENSORTYPE dtype){
   
    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
        tanh_kernel<<<blocks, threads>>>((float*)input, size);
    else
        throw std::runtime_error("tanh only implemented for float and bfloat16");
}

#endif