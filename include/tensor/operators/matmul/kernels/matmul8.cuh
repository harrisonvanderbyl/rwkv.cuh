#ifndef MATMUL8_CUH
#define MATMUL8_CUH

#include "tensor/operators/matmul/kernels/globals.cuh"
#include <cuda_bf16.h>
#ifndef jsplit
#define jsplit 16
#define tsplit 32
#endif

template <int bbtsplit>
__global__ void kernelc_mm8_one_float4(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y,
    unsigned long long tokenlength)

{

    const unsigned long long j0 = (threadIdx.x) * jsplit;

    const unsigned long long k0 = blockIdx.x * tsplit;

    const unsigned long long token = blockIdx.y * bbtsplit * 4;

    const auto *xk = (x + token * INPUTSIZE);

    float4 sum[bbtsplit] = {{0.0f}};

#pragma unroll
    for (unsigned long long j = 0; j < jsplit; j++)
    {
#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit * 4; i += 4)
        {
            sum[i].x += (xk[(j0 + j) + i * INPUTSIZE]);
            sum[i].y += (xk[(j0 + j) + i * INPUTSIZE + INPUTSIZE]);
            sum[i].z += (xk[(j0 + j) + i * INPUTSIZE + 2 * INPUTSIZE]);
            sum[i].w += (xk[(j0 + j) + i * INPUTSIZE + 3 * INPUTSIZE]);
        }
    }

#pragma unroll
    for (unsigned long long lk = 0; lk < tsplit; ++lk)
    {
        // permutation
        const auto k = (lk + threadIdx.y * tsplit + jsplit) % tsplit;

        float4 y_local[bbtsplit] = {{0.0f}};
        auto off = o[k0 + k];
        const auto *wk = (w + (k + k0) * INPUTSIZE);

#pragma unroll
        for (unsigned long long j = 0; j < jsplit; j++)
        {
            auto const mval = float(wk[(j0 + j)]);

#pragma unroll
            for (unsigned long long i = 0; i < bbtsplit * 4; i += 4)
            {
                y_local[i].x += (xk[(j0 + j) + i * INPUTSIZE] * mval);
                y_local[i].y += (xk[(j0 + j) + i * INPUTSIZE + INPUTSIZE] * mval);
                y_local[i].z += (xk[(j0 + j) + i * INPUTSIZE + 2 * INPUTSIZE] * mval);
                y_local[i].w += (xk[(j0 + j) + i * INPUTSIZE + 3 * INPUTSIZE] * mval);
            }
        }

#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit * 4; i += 4)
        {
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE, y_local[i].x * r[k0 + k] + off * sum[i].x);
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE + OUTPUTSIZE, y_local[i].y * r[k0 + k] + off * sum[i].y);
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE + 2 * OUTPUTSIZE, y_local[i].z * r[k0 + k] + off * sum[i].z);
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE + 3 * OUTPUTSIZE, y_local[i].w * r[k0 + k] + off * sum[i].w);
        }
    }
}

template <int bbtsplit>
__global__ void kernelc_mm8_one_float3(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y,
    unsigned long long tokenlength)

{

    const unsigned long long j0 = (threadIdx.x) * jsplit;

    const unsigned long long k0 = blockIdx.x * tsplit;

    const unsigned long long token = blockIdx.y * bbtsplit * 3;

    const auto *xk = (x + token * INPUTSIZE);

    float3 sum[bbtsplit] = {{0.0f}};

#pragma unroll
    for (unsigned long long j = 0; j < jsplit; j++)
    {
#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit * 3; i += 3)
        {
            sum[i].x += (xk[(j0 + j) + i * INPUTSIZE]);
            sum[i].y += (xk[(j0 + j) + i * INPUTSIZE + INPUTSIZE]);
            sum[i].z += (xk[(j0 + j) + i * INPUTSIZE + 2 * INPUTSIZE]);
        }
    }

#pragma unroll
    for (unsigned long long lk = 0; lk < tsplit; ++lk)
    {
        // permutation
        const auto k = (lk + threadIdx.y * tsplit + jsplit) % tsplit;

        float3 y_local[bbtsplit] = {{0.0f}};
        auto off = o[k0 + k];
        const auto *wk = (w + (k + k0) * INPUTSIZE);

#pragma unroll
        for (unsigned long long j = 0; j < jsplit; j++)
        {
            auto const mval = float(wk[(j0 + j)]);

#pragma unroll
            for (unsigned long long i = 0; i < bbtsplit * 3; i += 3)
            {
                y_local[i].x += (xk[(j0 + j) + i * INPUTSIZE] * mval);
                y_local[i].y += (xk[(j0 + j) + i * INPUTSIZE + INPUTSIZE] * mval);
                y_local[i].z += (xk[(j0 + j) + i * INPUTSIZE + 2 * INPUTSIZE] * mval);
            }
        }

#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit * 3; i += 3)
        {
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE, y_local[i].x * r[k0 + k] + off * sum[i].x);
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE + OUTPUTSIZE, y_local[i].y * r[k0 + k] + off * sum[i].y);
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE + 2 * OUTPUTSIZE, y_local[i].z * r[k0 + k] + off * sum[i].z);
        }
    }
}

template <int bbtsplit>
__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y,
    unsigned long long tokenlength)

{

    const unsigned long long j0 = (threadIdx.x) * jsplit;

    const unsigned long long k0 = blockIdx.x * tsplit;

    const unsigned long long token = blockIdx.y * bbtsplit * 3;

    const auto *xk = (x + token * INPUTSIZE);

    float sum[bbtsplit] = {0.0f};

#pragma unroll
    for (unsigned long long j = 0; j < jsplit; j++)
    {
#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit; i++)
        {
            sum[i] += (xk[(j0 + j) + i * INPUTSIZE]);
        }
    }

#pragma unroll
    for (unsigned long long lk = 0; lk < tsplit; ++lk)
    {
        // permutation
        const auto k = (lk + threadIdx.y * tsplit + jsplit) % tsplit;

        float y_local[bbtsplit] = {0.0f};
        auto off = o[k0 + k];
        const auto *wk = (w + (k + k0) * INPUTSIZE);

#pragma unroll
        for (unsigned long long j = 0; j < jsplit; j++)
        {
            auto const mval = float(wk[(j0 + j)]);

#pragma unroll
            for (unsigned long long i = 0; i < bbtsplit; i += 1)
            {
                y_local[i] += (xk[(j0 + j) + i * INPUTSIZE] * mval);
            }
        }

#pragma unroll
        for (unsigned long long i = 0; i < bbtsplit; i += 1)
        {
            atomicAdd(y + OUTPUTSIZE * token + k + k0 + i * OUTPUTSIZE, y_local[i] * r[k0 + k] + off * sum[i]);
        }
    }
}

void matmul8_cuda_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

    //     dim3 blockSize(32);
    //     dim3 gridSize(1024);

    //    auto jsplit = INSHAPE/blockSize.x;
    //     auto tsplit = OUTSHAPE/gridSize.x;
    if (BBT == 1)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, 1, 1);
        kernelc_mm8_one<1><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT == 2)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, 1, 1);
        kernelc_mm8_one<2><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT == 3)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, 1, 1);
        kernelc_mm8_one_float3<1><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT == 4)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, 1, 1);
        kernelc_mm8_one_float4<1><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT % 8 == 0)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, BBT / 8, 1);
        kernelc_mm8_one_float4<2><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }
    if (BBT % 4 == 0)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, BBT / 4, 1);
        kernelc_mm8_one_float4<1><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT % 6 == 0)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, BBT / 6, 1);
        kernelc_mm8_one_float3<2><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    if (BBT % 3 == 0)
    {
        dim3 blockSize(INSHAPE / jsplit, 1, 1);
        dim3 gridSize(OUTSHAPE / tsplit, BBT / 3, 1);
        kernelc_mm8_one_float3<1><<<gridSize, blockSize>>>(
            INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
        return;
    }

    dim3 blockSize(INSHAPE / jsplit, 1, 1);
    dim3 gridSize(OUTSHAPE / tsplit, BBT, 1);
    kernelc_mm8_one<1><<<gridSize, blockSize>>>(
        INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C, BBT);
    return;
}

#endif // MATMUL8_CUH