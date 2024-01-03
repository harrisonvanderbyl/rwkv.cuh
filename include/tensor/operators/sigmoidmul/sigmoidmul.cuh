#ifndef SIGMOIDMUL_CUH
#define SIGMOIDMUL_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

template <typename T>
__global__ void sigmoidmul_kernel(T *a, T *b, T *residual, T *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = T(float(b[idx]) / (1.0f + exp(-float(a[idx]))))+residual[idx];
    }
}

void sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype){
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
        sigmoidmul_kernel<<<blocks, threads>>>((float*)input, (float*)other, (float*)residual, (float*)output, size);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        sigmoidmul_kernel<<<blocks, threads>>>((bfloat16*)input, (bfloat16*)other, (bfloat16*)residual, (bfloat16*)output, size);
    else
        throw std::runtime_error("sigmoidmul only implemented for float and bfloat16");
}

#endif