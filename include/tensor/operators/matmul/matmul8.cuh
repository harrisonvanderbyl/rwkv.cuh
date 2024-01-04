#ifndef __MATMULFP_H__
#define __MATMULFP_H__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tensor/tensor.h"

__global__ void matmulfp_kernal(float* A, float*B, float*C, size_t INSHAPE, size_t OUTSHAPE){
    size_t bbt = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    float out = A[i * INSHAPE + k] * B[bbt * INSHAPE + k];

    atomicAdd(C + bbt * OUTSHAPE + i, out);
}

__global__ void matmulfp_kernal(__nv_bfloat16*  A, __nv_bfloat16* B, __nv_bfloat16* C, size_t INSHAPE, size_t OUTSHAPE, size_t CHUNKSIZE){
    size_t bbt = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    __nv_bfloat16 acc = 0;

    for (size_t j = 0; j < CHUNKSIZE; j++)
    {
        acc += A[i * INSHAPE + k*CHUNKSIZE + j] * B[bbt * INSHAPE + k*CHUNKSIZE + j];
    }


    atomicAdd(C + bbt * OUTSHAPE + i, acc);
}

template <typename T>
__global__ void wkvatt(size_t TT, size_t CH, T *kk, T *vv, T *rr, T *ww, T *uu, float *ss, T *out, size_t H)
{

    // bb is batch
    // hh is head
    size_t bb = blockIdx.x ;
    size_t hh = threadIdx.x;
    // 1d
    uint32_t bsize = H * TT * CH;

    // 1d tensor
    uint32_t tsize = H * CH;
    // 2d tensor
    uint32_t ttsize = H * CH * CH;

    // 1d
    uint32_t hsize = CH;
    // 2d
    uint32_t hhsize = CH * CH;

    for (uint32_t t = 0; t < TT; t++)
    {
        for (uint32_t i = 0; i < CH; i++)
        {
            auto btimeoffset = bb * bsize;
            auto timeoffset = btimeoffset + t * tsize;
            auto bbhsize = bb * ttsize;

            auto hoffset = hh * hsize;
            auto bhofseti = timeoffset + hoffset;
            auto bbhhsize = bbhsize + hh * hhsize;

            uint32_t iind = bhofseti + i;
            auto hoffseti = hoffset + i;
            auto bbhhofseti = bbhhsize + i * hsize;

            float kkk = float(kk[iind]);
            float uuu = float(uu[hoffseti]);
            float rrr = float(rr[iind]);
            float www = float(ww[hoffseti]);

            for (uint32_t j = 0; j < CH; j += 1)
            {
                uint32_t jind = bhofseti + j;
                uint32_t sind = bbhhofseti + j;

                // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                float vvv = float(vv[jind]);
                float sss = ss[sind];

                // multiply kkk and vvv
                auto atu = (vvv * kkk);

                // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                auto sssatuuuu = (atu * uuu + sss);

                out[jind] += T(sssatuuuu * rrr);

                ss[sind] = sss * www + atu;
            }
        }
    }
}

void matmul_cuda_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE,TENSORTYPE dtype){
    size_t CHUNKSIZE = 16;
    dim3 dimBlock(1, 8, 16);
    dim3 dimGrid(BBT, OUTSHAPE/8, (INSHAPE/CHUNKSIZE)/16);
    if (dtype == TENSORTYPE::kFLOAT_32)
        matmulfp_kernal<<<dimGrid, dimBlock>>>((float *)A, (float *)B, (float *)C, INSHAPE, OUTSHAPE);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        matmulfp_kernal<<<dimGrid, dimBlock>>>((__nv_bfloat16 *)A, (__nv_bfloat16 *)B, (__nv_bfloat16 *)C, INSHAPE, OUTSHAPE, CHUNKSIZE);
    else
        throw std::runtime_error("matmul not implemented for this dtype");
}

void  wkv5_cuda_kernel(void* kk, void* vv, void* ww, void* uu, void* rr, void* ss, void* out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype){
    dim3 dimBlock(H);
    dim3 dimGrid(B);
    if (dtype == TENSORTYPE::kFLOAT_32)
        wkvatt<<<dimGrid, dimBlock>>>(T, C / H, (float *)kk, (float *)vv, (float *)rr, (float *)ww, (float *)uu, (float *)ss, (float *)out, H);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        wkvatt<<<dimGrid, dimBlock>>>(T, C / H, (bfloat16 *)kk, (bfloat16 *)vv, (bfloat16 *)rr, (bfloat16 *)ww, (bfloat16 *)uu, (float *)ss, (bfloat16 *)out, H);
    else
        throw std::runtime_error("wkv5 not implemented for this dtype");
}

#endif