#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

#define CUNORMTHREADS 32
#define CUNORMBLOCKS 1

__global__ void layernorm(float* input, float* output, float* weight, float* bias, size_t size, size_t lastshape, size_t headshape, float eps = 1e-5){
    size_t thread = threadIdx.x;
    size_t hh = blockIdx.x;

    auto start = hh * headshape;
    auto wb = start%lastshape;
    weight += wb;
    bias += wb;
    input += start;
    output += start;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (size_t i = thread; i < headshape; i+=CUNORMTHREADS){
        sum += input[i];
        sumsq += input[i] * input[i];
    }

    for (int i = 1; i < warpSize; i *= 2)
    {
        sum += __shfl_xor_sync(0xffffffff, sum, i);
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, i);
    }
    
   
    float mean = sum / headshape;
    float var = sumsq / headshape - mean * mean;

    float invstd = 1.0f / sqrt(var + eps);

    for (size_t i = thread; i < headshape; i+=CUNORMTHREADS){
        output[i] = (input[i] - mean) * invstd * weight[i] + bias[i];
    }
}



void normalize_cuda_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype){

        

        // batchsize
        size_t blocks = size/headshape;
        auto gridsize = dim3(blocks,1,1);
        auto headsperblock = dim3(CUNORMTHREADS,1,1);

       
        if (dtype == TENSORTYPE::kFLOAT_32)
            layernorm<<<gridsize, headsperblock>>>((float*)input, (float*)output, (float*)weight, (float*)bias, size, lastshape, headshape, eps);
        else
            throw std::runtime_error("Unsupported datatype for normalize, only float32 and bfloat16 are supported on CUDA");
        
}
#endif // NORMALIZE_CUH