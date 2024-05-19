#ifndef SIGMOIDMUL_CUH
#define SIGMOIDMUL_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor/tensor.h"

__global__ void sigmoidmul_kernel(float *a, float *b, float *residual, float *c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = (b[idx]) / (1.0f+ exp(-a[idx])) +residual[idx];
    }
}


void sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, size_t size, TENSORTYPE dtype){
   
    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
        sigmoidmul_kernel<<<blocks, threads>>>((float*)input, (float*)other, (float*)residual, (float*)output, size);
    else
        throw std::runtime_error("sigmoidmul only implemented for float and bfloat16");
}

#endif