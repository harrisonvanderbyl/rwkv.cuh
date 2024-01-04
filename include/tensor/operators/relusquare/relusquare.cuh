#ifndef RELUSQUARE_CUH
#define RELUSQUARE_CUH
#include "tensor/tensor.h"
#include "cuda_runtime.h"

template <typename T>
__global__ void relusquare_kernel(T *a, T *b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (float(a[idx]) > 0) {
            b[idx] = a[idx] * a[idx];
        } else {
            b[idx] = 0;
        }
    }
}

void relusquare_cuda_kernel(void* input, void* output, int size, TENSORTYPE dtype){
    int block_size = 1024;
    int grid_size = (size + block_size - 1) / block_size;
    if (dtype == TENSORTYPE::kFLOAT_32) {
        relusquare_kernel<float><<<grid_size, block_size>>>((float*)input, (float*)output, size);
    } else if (dtype == TENSORTYPE::kBFLOAT_16) {
        relusquare_kernel<bfloat16><<<grid_size, block_size>>>((bfloat16*)input, (bfloat16*)output, size);
    } else{
        printf("not support dtype for cuda\n");
        exit(-1);
    }
}

#endif // RELUSQUARE_CUH