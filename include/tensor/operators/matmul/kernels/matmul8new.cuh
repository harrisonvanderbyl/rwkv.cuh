#ifndef MATMUL8_CUH
#define MATMUL8_CUH

#include "tensor/operators/matmul/kernels/globals.cuh"
#include <cuda_bf16.h>
#ifndef jsplit
#define jsplit 1024
#define tsplit 64
#endif

__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    unsigned long long tokenlength,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y)
{

    const unsigned long long outindex = blockIdx.z* (jsplit) + threadIdx.x;
    const unsigned long long token = outindex / OUTPUTSIZE;
    const unsigned long long out = outindex % OUTPUTSIZE;

    const unsigned long long start = blockIdx.x*tsplit;

    
    auto range = r[out];
    auto off = o[out]/range;
    
    const auto *wk = (w + (out)*INPUTSIZE);


    __shared__ float inmat[tsplit];

    if (threadIdx.x < tsplit)
    {
        inmat[threadIdx.x] = x[token * INPUTSIZE + (start + threadIdx.x)];
    }
    __syncthreads();
    
   

    // y_local[threadIdx.x] += x[token * INPUTSIZE + (start + threadIdx.y)] * mval;
    auto yylocal = 0.0f;
    #pragma unroll
    for (int i = 0; i < tsplit; i++)
    {

        const float mval = off + (float)wk[start + i];
        yylocal += inmat[i] * mval;
    }
    
      

    atomicAdd(&y[outindex],  range * yylocal);
}

void matmul8_cuda_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

    //     dim3 blockSize(32);
    //     dim3 gridSize(1024);

    //    auto jsplit = INSHAPE/blockSize.x;
    //     auto tsplit = OUTSHAPE/gridSize.x;

    dim3 blockSize(jsplit, 1, 1);
    dim3 gridSize(INSHAPE/tsplit, 1, (OUTSHAPE*BBT)/ (jsplit) );
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        OUTSHAPE, INSHAPE, BBT, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C);
    return;
}

#endif // MATMUL8_CUH