#ifndef MATMUL8_CUH
#define MATMUL8_CUH

#include "tensor/operators/matmul/kernels/globals.cuh"
#include <cuda_bf16.h>
#define jsplit 32
#define tsplit 16

__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const bfloat16 *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y,
    unsigned long long tokenlength)

{

    const unsigned long long j0 = (threadIdx.x) * jsplit;

    const unsigned long long k0 = blockIdx.x * tsplit;

#pragma unroll
    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        const auto *xk = (x + token * INPUTSIZE * 2);

        float sum = 0.0f;

#pragma unroll
        for (unsigned long long j = 0; j < jsplit; j++)
        {
            sum += float(x[token * INPUTSIZE * 2 + (j0 + j) * 2 + 1]);
        }

#pragma unroll
        for (unsigned long long k = 0; k < tsplit; ++k)
        {

            auto y_local = 0.0f;
            auto off = o[k0 + k];
            const auto *wk = (w + (k + k0) * INPUTSIZE);

#pragma unroll
            for (unsigned long long j = 0; j < jsplit; j++)
            {
                y_local += float(xk[(j0 + j) * 2 + 1] * __ushort2bfloat16_rn(wk[(j0 + j)]));
            }

            atomicAdd(y + OUTPUTSIZE * token + k + k0, y_local * r[k0 + k] + off * sum);
        }
    }
}

void matmul8_cuda_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

    //     dim3 blockSize(32);
    //     dim3 gridSize(1024);

    //    auto jsplit = INSHAPE/blockSize.x;
    //     auto tsplit = OUTSHAPE/gridSize.x;

    dim3 blockSize(INSHAPE / jsplit);
    dim3 gridSize(OUTSHAPE / tsplit);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        INSHAPE, OUTSHAPE, (bfloat16 *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
}

#endif // MATMUL8_CUH