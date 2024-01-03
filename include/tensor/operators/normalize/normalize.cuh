#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include <cuda_runtime.h>
#include "tensor/tensor.h"

template <typename T>
__global__ void layernorm(T* input, T* output, T* weight, T* bias, size_t size, size_t lastshape, size_t headshape, float eps = 1e-5){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        size_t head = idx / headshape;
        size_t offset = head * headshape;
        float mean = 0;
        float var = 0;
        for (size_t i = offset; i < offset + headshape; i++){
            mean += float(input[i]);
        }
        mean /= headshape;
        for (size_t i = offset; i < offset + headshape; i++){
            var += (float(input[i]) - mean) * (float(input[i]) - mean);
        }
        var /= headshape;
        for (size_t i = offset; i < offset + headshape; i++){
            output[i] = ((float(input[i]) - mean) / sqrt(var + eps)) * float(weight[i%lastshape]) + float(bias[i%lastshape]);
        }
    }
}



void normalize_cuda_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype){

        

        // one instance per head
        size_t threads = size / headshape;

        // one thread per element
        size_t blocks = headshape;

       
        if (dtype == TENSORTYPE::kFLOAT_32)
            layernorm<<<blocks, threads>>>((float*)input, (float*)output, (float*)weight, (float*)bias, size, lastshape, headshape, eps);
        else if (dtype == TENSORTYPE::kBFLOAT_16)
            layernorm<<<blocks, threads>>>((bfloat16*)input, (bfloat16*)output, (bfloat16*)weight, (bfloat16*)bias, size, lastshape, headshape, eps);
        else
            throw std::runtime_error("Unsupported datatype for normalize, only float32 and bfloat16 are supported on CUDA");
        
}
#endif // NORMALIZE_CUH