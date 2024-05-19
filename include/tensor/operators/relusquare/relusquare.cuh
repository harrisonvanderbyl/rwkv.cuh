#ifndef RELUSQUARE_CUH
#define RELUSQUARE_CUH
#include "tensor/tensor.h"
#include "cuda_runtime.h"

__global__ void relusquare_kernel(float *a, float *b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (float(a[idx]) > 0) {
            b[idx] = a[idx] * a[idx];
        } else {
            b[idx] = 0;
        }
    }
}

void relusquare_cuda_kernel(void* input, void* output, size_t size, TENSORTYPE dtype){
    size_t block_size = 512;
    size_t grid_size = (size + block_size - 1) / block_size;
    if (dtype == TENSORTYPE::kFLOAT_32) {
        relusquare_kernel<<<grid_size, block_size>>>((float*)input, (float*)output, size);
    } else{
        printf("not support dtype for cuda\n");
        exit(-1);
    }
}

#endif // RELUSQUARE_CUH