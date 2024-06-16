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

struct bfloat1624;

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
  __device__ void loadFromGmem8(int N, int K, const float *A, const float *maxA, const uint8_t *B, const uint8_t *OB, bfloat1624 *range, bfloat1624 *off,
                                float *As, float4 *Bs, int innerRowA, int innerColA,
                                int innerRowB, int innerColB)
  {
#pragma unroll
    for (uint offset = 0; offset < BM; offset += rowStrideA)
    {
      if (
          A + (innerRowA + offset) * (K * 1) + innerColA * 4 < maxA)
      {
        const auto tmp = (float4 *)(A + (innerRowA + offset) * (K * 1) + innerColA * 4);

        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp->x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp->y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp->z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp->w;
      }
    }

#pragma unroll
    for (uint offset = 0; offset < BK; offset += rowStrideB)
    {
      unsigned long int start = (B + (innerRowB + offset) * N + innerColB * 4) - OB;
      auto row = start % N;
      auto col = start / N;
      start = col + row * K;

      auto bbs = Bs + (innerRowB + offset) * BN4 + innerColB;

      auto bbsx = bfloat1624({__float22bfloat162_rn(make_float2(OB[start], OB[start + K * 1])), __float22bfloat162_rn(make_float2(OB[start + K * 2], OB[start + K * 3]))});
      bbsx.x = bbsx.x * range[innerColB].x + off[innerColB].x;
      bbsx.y = bbsx.y * range[innerColB].y + off[innerColB].y;
      bbs[0].x = bbsx.x.x;
      bbs[0].y = bbsx.x.y;
      bbs[0].z = bbsx.y.x;
      bbs[0].w = bbsx.y.y;

      // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
      //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
  }

  __device__ void
  processFromSmem8(float *regM, float4 *regN, float4 *threadResults, const float *As,
                   const float4 *Bs, const uint warpRow, const uint warpCol,
                   const uint threadRowInWarp, const uint threadColInWarp, const uint M)
  {
    // auto regMf4 = reinterpret_cast<float4 *>(regM);
    // auto As4 = reinterpret_cast<const float4 *>(As);
#pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // populate registers for whole warptile

#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
      {
#pragma unroll
        for (uint i = 0; i < TM; ++i)
        {
          regM[wSubRowIdx * TM + i] =
              As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
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
              auto rnn = regM[wSubRowIdx * TM + resIdxM];

              // UFMAFF(rn, , ((unsigned int *)iiv));
              iiv->x += rn->x * rnn;
              iiv->y += rn->y * rnn;
              iiv->z += rn->z * rnn;
              iiv->w += rn->w * rnn;
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
    sgemmWarptiling8(int M, int N, int K, float *A, uint8_t *B, bfloat1624 *range, bfloat1624 *off,
                     float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const float *maxc = C + M * N;
  const float *maxA = A + M * K;
  const uint8_t *OB = B;

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
  __shared__ float4 Bs[BK * BN4];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
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
  float4 threadResults4[WMITER * TM * WNITER * TN4];
  for (uint yy = 0; yy < WMITER * TM * WNITER * TN4; yy++)
  {
    threadResults4[yy] = make_float4(0, 0, 0, 0);
  }
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float4 regN[WNITER * TN4];
  for (uint yy = 0; yy < WMITER * TN4; yy++)
  {
    regN[yy] = make_float4(0, 0, 0, 0);
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    wt::loadFromGmem8(
        N, K, A, maxA, B, OB, range, off, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem8(regM, regN, threadResults4, As, Bs, warpRow, warpCol,
                         threadRowInWarp, threadColInWarp, M);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

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

          auto iix = C + (cRow * BM + warpRow * WM + threadRowInWarp * TM + wSubRowIdx * WSUBM + resIdxM) * N + cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN * 4;
          if (iix < maxc)
          {
            auto iiy = (float4 *)(((float *)threadResults4) + (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN * 4);

            auto iixh = reinterpret_cast<float4 *>(iix);

            iixh->x += iiy->x;
            iixh->y += iiy->y;
            iixh->z += iiy->z;
            iixh->w += iiy->w;
          }
        }
      }
    }
  }
}

void runSgemmWarptiling8(int M, int N, int K, float *A, uint8_t *B, bfloat1624 *range, bfloat1624 *off, float *C)
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

#define jsplit 1024
#define tsplit 64

template <int INSS>
__global__ void kernelc_mm8_one(
    const unsigned long long INPUTSIZE,
    const unsigned long long OUTPUTSIZE,
    float *x,
    uint8_t *w,
    __nv_bfloat162 *r,
    __nv_bfloat162 *o,
    float *y)
{

  const unsigned long long outstart = blockIdx.x * (tsplit);

  const unsigned long long instart = threadIdx.x * INSS;

  r += blockIdx.x * 32;
  o += blockIdx.x * 32;
  w += outstart * INPUTSIZE + instart ;
  x += instart;
  y += outstart;
  auto ystart = (float2 *)y;
  auto warppos = threadIdx.x % warpSize;
  auto warp = threadIdx.x / warpSize;
  // y_local[threadIdx.x] += x[token * INPUTSIZE + (start + threadIdx.y)] * mval;

  // bfloat16 __shared__ xinp[jsplit];

  // bfloat16 xsum = ((bfloat16 *)(x + threadIdx.x))[1];
  // xinp[threadIdx.x] = xsum;

  // for (int i = 1; i < warpSize; i *= 2)
  //   xsum += __shfl_xor_sync(-1, xsum, i);
  // bfloat16 xsum = __shfl_sync(0xffffffff, sumate, 0);
  

 __shared__ __nv_bfloat162 hotspace[32][32];

  __nv_bfloat16 xsum = __float2bfloat16(0.0);

#pragma unroll
  for (auto j = 0; j < (INSS); j += 1)
    {
      auto xinp = __float2bfloat16(x[j]);
       xsum += xinp;

    }

__nv_bfloat162 xsumx = __nv_bfloat162(xsum,xsum);

  
size_t end = OUTPUTSIZE - outstart;
#pragma unroll
  for (int i = 0; i < 32 && i < end; i+=1)
  {

    __nv_bfloat162 xout = make_bfloat162(__float2bfloat16(0.0),__float2bfloat16(0.0));
   

#pragma unroll
    for (auto j = 0; j < (INSS); j += 1)
    {
      // asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}" : "=r"(*out) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)), "r"(__BFLOAT162_TO_CUI(c)));
      auto nb = __float22bfloat162_rn(make_float2(w[j], w[j + INPUTSIZE]));
      auto xinp = __float2bfloat162_rn(x[j]);
      asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}"
          : "=r"(__BFLOAT162_TO_CUI(xout)) : "r"(__BFLOAT162_TO_CUI(xinp)), "r"(__BFLOAT162_TO_CUI(nb)), "r"(__BFLOAT162_TO_CUI(xout)));
      // xsum += xinp;
    }

    auto xouts = __hfma2(xout, *r, *o * (xsumx));

    for (int ij = 1; ij < warpSize; ij *= 2)
      xouts += __shfl_xor_sync(-1, xouts, ij);

    if (warppos == 0)
    {
      // atomicAdd(ystart + i * 2 + 1, xouts);
      hotspace[i][warp] = xouts;
    }

    

    w += INPUTSIZE*2;
    r += 1;
    o += 1;
  }

  __syncthreads();

  // if (warp  < tsplit/2)
  // {
  auto xouts = hotspace[warp][warppos];
  for (int i = 1; i < warpSize; i *= 2)
    xouts += __shfl_xor_sync(-1, xouts, i);
    
  if (warppos == 0 && warp < OUTPUTSIZE-outstart)
  {
    auto outt = __bfloat1622float2(xouts);
    atomicAdd(&ystart[warp].x, outt.x);
    atomicAdd(&ystart[warp].y, outt.y);
   
    
  }
  // }
}

void matmul8_cuda_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

  if (BBT != 1 || (OUTSHAPE%tsplit != 0))
  {
    runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uint8_t *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
  }
  else
  {

    dim3 gridSize(OUTSHAPE / (tsplit), 1, 1);
    // assert(tsplit%OUTSHAPE == 0);
    
    dim3 blockSize(jsplit, 1, 1);

    if (INSHAPE == 2048)
    {
      kernelc_mm8_one<2048 / jsplit><<<gridSize, blockSize>>>(
          INSHAPE, OUTSHAPE, (float *)B, A, (__nv_bfloat162 *)Ar, (__nv_bfloat162 *)Ao, (float *)C);
    }
    else if (INSHAPE == 2560)
    {
      // kernelc_mm8_one<2560/jsplit><<<gridSize, blockSize>>>(
      //     INSHAPE, OUTSHAPE, (float *)B, A, (__nv_bfloat162 *)Ar, (__nv_bfloat162 *)Ao, (float *)C);
      runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uint8_t *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
    }
    else if (INSHAPE == 8960)
    {
      // kernelc_mm8_one<8960/jsplit><<<gridSize, blockSize>>>(
      //     INSHAPE, OUTSHAPE, (float *)B, A, (__nv_bfloat162 *)Ar, (__nv_bfloat162 *)Ao, (float *)C);
      runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uint8_t *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
    }
    else if (
        INSHAPE == 7168)
    {
      runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uint8_t *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
    }
    else if (
        INSHAPE == 4096)
    {
      kernelc_mm8_one<4096 / jsplit><<<gridSize, blockSize>>>(
          INSHAPE, OUTSHAPE, (float *)B, A, (__nv_bfloat162 *)Ar, (__nv_bfloat162 *)Ao, (float *)C);
    }
    else if (
        INSHAPE == 14336)
    {
      kernelc_mm8_one<14336 / jsplit><<<gridSize, blockSize>>>(
          INSHAPE, OUTSHAPE, (float *)B, A, (__nv_bfloat162 *)Ar, (__nv_bfloat162 *)Ao, (float *)C);
    }
    else
    {
      runSgemmWarptiling8(BBT, OUTSHAPE, INSHAPE, (float *)B, (uint8_t *)A, (bfloat1624 *)Ar, (bfloat1624 *)Ao, (float *)C);
    }
  }
  // max size 1024
}

