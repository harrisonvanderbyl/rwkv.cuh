#ifndef SWISHMUL_CUH
#define SWISHMUL_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

template <typename T>
__global__ void swishmul_kernel(T *a, T *b, T *c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = T((float(a[idx]) * float(b[idx])) / (1.0f + exp(-float(a[idx]))));
    }
}

void swishmul_cuda_kernel(void* input, void* other, void* output, size_t size, TENSORTYPE dtype){
    size_t threads = 512;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
        swishmul_kernel<<<blocks, threads>>>((float*)input, (float*)other, (float*)output, size);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        swishmul_kernel<<<blocks, threads>>>((bfloat16*)input, (bfloat16*)other, (bfloat16*)output, size);
    else
        throw std::runtime_error("swishmul only implemented for float and bfloat16");
}

#endif