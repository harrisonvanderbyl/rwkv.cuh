#ifndef tahn_CUH
#define tahn_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

template <typename T>
__global__ void tahn_kernel(T *a, T *c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        auto e2x = exp(2.0f * float(a[idx]));
        c[idx] = ((e2x - 1.0f) / (e2x + 1.0f));
    }
}

void tahn_cuda_kernel(void* input, void* output, size_t size, TENSORTYPE dtype){
    size_t threads = 512;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
        tahn_kernel<<<blocks, threads>>>((float*)input,  (float*)output, size);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        tahn_kernel<<<blocks, threads>>>((bfloat16*)input,  (bfloat16*)output, size);
    else
        throw std::runtime_error("tahn only implemented for float and bfloat16");
}

#endif