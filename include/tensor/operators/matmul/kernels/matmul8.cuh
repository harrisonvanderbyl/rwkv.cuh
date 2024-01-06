#ifndef MATMUL8_CUH
#define MATMUL8_CUH

#include "tensor/operators/matmul/kernels/globals.cuh"

template <typename DTYPE>
__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const DTYPE *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y,
    unsigned long long tokenlength)

{

    const unsigned long long j0 = (threadIdx.x) * MM8_ONE_JSPLIT;
    const unsigned long long j1 = (threadIdx.x + 1) * MM8_ONE_JSPLIT;
    
    const unsigned long long k0 = blockIdx.x * MM8_ONE_TILE;
    const unsigned long long k1 = (blockIdx.x + 1) * MM8_ONE_TILE;
        

    #pragma unroll
    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        

        #pragma unroll
        for (unsigned long long k = k0; k < k1; ++k)
        {
        
            float y_local = 0;
            float off = o[k]/r[k];

            #pragma unroll
            for (unsigned long long j = j0; j < j1; ++j)
            {
                y_local += float(x[token * INPUTSIZE + j]) * ((w[k * INPUTSIZE + j] + off));
            }

            atomicAdd(y + OUTPUTSIZE * token + k , y_local * r[k] );
        }
    }
}

#endif // MATMUL8_CUH