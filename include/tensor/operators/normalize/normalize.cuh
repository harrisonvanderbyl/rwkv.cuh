#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

template <typename T>
__global__ void layernorm(T* input, T* output, T* weight, T* bias, size_t size, size_t lastshape, size_t headshape, float eps = 1e-5){
    size_t bb = blockIdx.x;
    size_t hh = threadIdx.x;

    size_t start = bb*lastshape + hh*headshape;
    size_t end = start + headshape;

    float sum = 0;
    
    for (size_t i = start; i < end; i+=1){
        sum += float(input[i]);
    }
    float mean = sum / headshape;

    float vars = 0;
    for (size_t i = start; i < end; i+=1){
        vars += (float(input[i]) - mean) * (float(input[i]) - mean);
    }
    float var = vars / headshape;

    float invstd = (1.0f) / sqrt(var + eps);

    for (size_t i = start; i < end; i+=1){
        output[i] = (input[i] - T(mean)) * T(invstd) * weight[hh*headshape + i-start] + bias[hh*headshape + i-start];
    }
}



void normalize_cuda_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype){

        

        // batchsize
        size_t blocks = size/lastshape;
        size_t heads = size/headshape;
        size_t headsperblock = heads/blocks;

       
        if (dtype == TENSORTYPE::kFLOAT_32)
            layernorm<<<blocks, headsperblock>>>((float*)input, (float*)output, (float*)weight, (float*)bias, size, lastshape, headshape, eps);
        else if (dtype == TENSORTYPE::kBFLOAT_16)
            layernorm<<<blocks, headsperblock>>>((bfloat16*)input, (bfloat16*)output, (bfloat16*)weight, (bfloat16*)bias, size, lastshape, headshape, eps);
        else
            throw std::runtime_error("Unsupported datatype for normalize, only float32 and bfloat16 are supported on CUDA");
        
}
#endif // NORMALIZE_CUH