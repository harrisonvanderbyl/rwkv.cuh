#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "tensor/operators/matmul/kernels/globals.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// uint8

#define BN 32U
#define BN4 8U
#define BM 32U
#define BM4 8U
#define BK 16U
#define BK4 4U
#define WN 32U
#define WN4 8U
#define WM 32U
#define WM4 8U
#define WNITER 1U
#define TN 4U
#define TN4 1U
#define TM 4U
#define TM4 1U
#define NUM_THREADS (BN / WN) * (BM / WM) * 32
#define rowStrideA (NUM_THREADS * 4) / BK
#define rowStrideB NUM_THREADS / (BN4)
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))

#define WSUBM WM / WMITER   // 64/2=32
#define WSUBN4 WN4 / WNITER // 16/2=8
#define WSUBN WN / WNITER   // 32/2=16
#define WSUBM4 WM4 / WMITER // 8/2=4
// const int WARPSIZE = 32; // warpSize is not constexpr

struct bfloat1624
{
  __nv_bfloat162 x = __float2bfloat162_rn(0.0f);
  __nv_bfloat162 y = __float2bfloat162_rn(0.0f);
};

// __fmaf_rn((a.x), (b.x), (c.x)), __fmaf_rn((a.y), (b.y), (c.y)), __fmaf_rn((a.z), (b.z), (c.z)), __fmaf_rn((a.w), (b.w), (c.w)) \

#define bf16fma(a, b, c) asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}" : "=r"(*c) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)), "r"(*c));
#define bf16fmaout(a, b, c, out) asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}" : "=r"(*out) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)), "r"(__BFLOAT162_TO_CUI(c)));

#define UFMAF(a, b, c, d)                                                   \
  bf16fmaout(__floats2bfloat162_rn(float(a.x), float(a.y)), b.x, c.x, (d)); \
  bf16fmaout(__floats2bfloat162_rn(float(a.z), float(a.w)), b.y, c.y, (d + 1));

#define UFMAFF(a, b, c)      \
  bf16fma((a->x), (b), (c)); \
  bf16fma((a->y), (b), (c + 1));

// host
namespace wt
{
  __device__ void loadFromGmem8(int N, int K, const float *A, const float *maxA, const uchar4 *B, bfloat1624 *range, bfloat1624 *off,
                                float *As, bfloat1624 *Bs, int innerRowA, int innerColA,
                                int innerRowB, int innerColB)
  {
#pragma unroll
    for (uint offset = 0; offset < BM; offset += rowStrideA)
    {
      if (
          A + (innerRowA + offset) * K + innerColA * 4 < maxA)
      {
        const auto tmp = (float4 *)(A + (innerRowA + offset) * K + innerColA * 4);

        asm("{.reg .b16 low;\n"
            "  cvt.rn.bf16.f32 low, %1;\n"
            "  mov.b32 %0, {low,low};}\n" : "=r"(*((unsigned int *)(As + (innerColA * 4 + 0) * BM + innerRowA + offset))) : "f"((tmp)->x));
        asm("{.reg .b16 low;\n"
            "  cvt.rn.bf16.f32 low, %1;\n"
            "  mov.b32 %0, {low,low};}\n" : "=r"(*((unsigned int *)(As + (innerColA * 4 + 1) * BM + innerRowA + offset))) : "f"((tmp)->y));
        asm("{.reg .b16 low;\n"
            "  cvt.rn.bf16.f32 low, %1;\n"
            "  mov.b32 %0, {low,low};}\n" : "=r"(*((unsigned int *)(As + (innerColA * 4 + 2) * BM + innerRowA + offset))) : "f"((tmp)->z));
        asm("{.reg .b16 low;\n"
            "  cvt.rn.bf16.f32 low, %1;\n"
            "  mov.b32 %0, {low,low};}\n" : "=r"(*((unsigned int *)(As + (innerColA * 4 + 3) * BM + innerRowA + offset))) : "f"((tmp)->w));
      }
    }

#pragma unroll
    for (uint offset = 0; offset < BK; offset += rowStrideB)
    {

      UFMAF(B[(innerRowB + offset) * N / 4 + innerColB], range[innerColB], off[innerColB], ((unsigned int *)(Bs + (innerRowB + offset) * BN4 + innerColB)));

      // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
      //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
  }

  __device__ void
  processFromSmem8(float *regM, bfloat1624 *regN, bfloat1624 *threadResults, const float *As,
                   const bfloat1624 *Bs, const uint warpRow, const uint warpCol,
                   const uint threadRowInWarp, const uint threadColInWarp, const uint M)
  {
    auto regMf4 = reinterpret_cast<float4 *>(regM);
    auto As4 = reinterpret_cast<const float4 *>(As);
#pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // populate registers for whole warptile

#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
      {
#pragma unroll
        for (uint i = 0; i < TM4; ++i)
        {
          regMf4[wSubRowIdx * TM4 + i] =
              As4[(dotIdx * BM4) + warpRow * WM4 + wSubRowIdx * WSUBM4 +
                  threadRowInWarp * TM4 + i];
        }
      }
#pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
      {
#pragma unroll
        for (uint i = 0; i < TN4; ++i)
        {
          regN[wSubColIdx * TN4 + i] = Bs[(dotIdx * BN4) + warpCol * WN4 + wSubColIdx * WSUBN4 + threadColInWarp * TN4 + i];
        }
      }

// execute warptile matmul
#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
      {
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
        {
// calculate per-thread results
#pragma unroll
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
          {
#pragma unroll
            for (uint resIdxN = 0; resIdxN < TN4; ++resIdxN)
            {
              auto iiv = threadResults + (wSubRowIdx * TM + resIdxM) * (WNITER * TN4) +
                         (wSubColIdx * TN4) + resIdxN;

              auto rn = regN + wSubColIdx * TN4 + resIdxN;

              UFMAFF(rn, regM[wSubRowIdx * TM + resIdxM], ((unsigned int *)iiv));
            }
          }
        }
      }
    }
  }

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling8(int M, int N, int K, float *A, uchar4 *B, bfloat1624 *range, bfloat1624 *off,
                     float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const float *maxc = C + M * N;
  const float *maxA = A + M * K;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ bfloat1624 Bs[BK * BN4];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN4;
  range += cCol * BN4;
  off += cCol * BN4;
  // Move C_ptr to warp's output tile
  

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step

  const uint innerRowA = threadIdx.x / (BK4);
  const uint innerColA = threadIdx.x % (BK4);
  const uint innerRowB = threadIdx.x / (BN4);
  const uint innerColB = threadIdx.x % (BN4);

  // allocate thread-local cache for results in registerfile
  bfloat1624 threadResults4[WMITER * TM * WNITER * TN4];
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  bfloat1624 regN[WNITER * TN4];

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    wt::loadFromGmem8(
        N, K, A, maxA, B, range, off, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem8(regM, regN, threadResults4, As, Bs, warpRow, warpCol,
                         threadRowInWarp, threadColInWarp, M);
    A += BK;      // move BK columns to right
    B += BK4 * N; // move BK rows down

    __syncthreads();
  }

  // float4 *threadResults4 = (float4 *)threadResults;

// write out the results
#pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
  {

#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
    {
#pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
      {

#pragma unroll
        for (uint resIdxN = 0; resIdxN < TN4; resIdxN += 1)
        {

          auto iix = C + (cRow * BM + warpRow * WM + threadRowInWarp * TM + wSubRowIdx * WSUBM + resIdxM) * N + cCol * BN + warpCol * WN + wSubColIdx * WSUBN  + threadColInWarp * TN + resIdxN * 4;
          if (iix < maxc)
          {
            auto iiy = (bfloat1624*)(((bfloat16*)threadResults4) + (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN*4);

            auto iixh = reinterpret_cast<bfloat1624 *>(iix);
            auto iixh2 = iixh + 1;
            iixh->x.x = iixh->x.y;
            iixh->x.y = iixh->y.y;
            iixh->y.x = iixh2->x.y;
            iixh->y.y = iixh2->y.y;


            auto a = __hadd2(iiy->x, iixh->x);
            auto b = __hadd2(iiy->y, iixh->y);
            iixh->x.y = a.x;
            iixh->y.y = a.y;
            iixh2->x.y = b.x;
            iixh2->y.y = b.y;
          }
        }
      }
    }
  }
}

void runSgemmWarptiling8(int M, int N, int K, float *A, uchar4 *B, bfloat1624 *range, bfloat1624 *off, float *C)
{
  // Settings for A100

  dim3 blockDim(NUM_THREADS);

  constexpr uint NUM_WARPS = NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) ==
                0);

  // warpsubtile in warptile
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

  static_assert((NUM_THREADS * 4) % BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((NUM_THREADS * 4) % BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  // static_assert(BN % (16 * TN) == 0,
  //               "BN must be a multiple of 16*TN to avoid quantization effects");
  // static_assert(BM % (16 * TM) == 0,
  //               "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemmWarptiling8<<<gridDim, blockDim>>>(M, N, K, A, B, range, off, C);

  cudaDeviceSynchronize();
  // get error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel failed");
  }
}

#define jsplit 64
#define tsplit 64

__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE, const unsigned long long OUTPUTSIZE,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    float *__restrict__ const y)
{

  const unsigned long long instart = threadIdx.x;

  const unsigned long long outstart = blockIdx.x;

  // y_local[threadIdx.x] += x[token * INPUTSIZE + (start + threadIdx.y)] * mval;
  auto yylocal = 0.0f;

  auto range = r[outstart];
  auto off = o[outstart];
  for (int j = 0; j < INPUTSIZE; j++)
  {

    yylocal += x[(instart + j)] * (off + range * ((float)w[(instart + j) * OUTPUTSIZE + (outstart)]));
  }
  atomicAdd(&y[outstart], yylocal);
}

void matmul8_cuda_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

  // if (BBT != 1)
  // {
  runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uchar4 *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
  // }
  // else{

  //   dim3 blockSize(1, 1, 1);
  //   dim3 gridSize(OUTSHAPE, 1, 1);
  //   kernelc_mm8_one<<<gridSize, blockSize>>>(
  //       INSHAPE, OUTSHAPE, (float *)B, A, (float *)Ar, (float *)Ao, (float *)C);

  // }
  // max size 1024
}
