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

// __fmaf_rn((a.x), (b.x), (c.x)), __fmaf_rn((a.y), (b.y), (c.y)), __fmaf_rn((a.z), (b.z), (c.z)), __fmaf_rn((a.w), (b.w), (c.w)) \

#define bf16fma(a, b, c) asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}" : "=r"(*c) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)), "r"(*c));
#define bf16fmaout(a, b, c, out) asm("{fma.rn.bf16x2 %0,%1,%2,%3;\n}" : "=r"(*out) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)), "r"(__BFLOAT162_TO_CUI(c)));

#define UFMAF(a, b, c, d)                                                   \
  bf16fmaout(__floats2bfloat162_rn(float(a.x), float(a.y)), b.x, c.x, (d)); \
  bf16fmaout(__floats2bfloat162_rn(float(a.z), float(a.w)), b.y, c.y, (d + 1));

#define UFMAFF(a, b, c)      \
  bf16fma((a->x), (b), (c)); \
  bf16fma((a->y), (b), (c + 1));

namespace wt
{

  __device__ void processFromSmem8(float *regM, float4 *regN, float4 *threadResults, const float *As,
                                   const float4 *Bs, const uint warpRow, const uint warpCol,
                                   const uint threadRowInWarp, const uint threadColInWarp, const uint M);
  // host

  template <typename T>
  __device__ void loadFromGmem(int N, int K, const float *A, const float *maxA, const T *B, const T *OB,
                               float *As, float4 *Bs, int innerRowA, int innerColA,
                               int innerRowB, int innerColB, int bmmshape)
  {
#pragma unroll
    for (uint offset = 0; offset < BM; offset += rowStrideA)
    {
      if (
          A + (innerRowA + offset) * (K * bmmshape) + innerColA * 4 < maxA)
      {
        const auto tmp = (float4 *)(A + (innerRowA + offset) * (K * bmmshape) + innerColA * 4);

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

      bbs[0] = make_float4(OB[start], OB[start + K * 1], OB[start + K * 2], OB[start + K * 3]);

      // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
      //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
  }
}
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
template <typename T>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float *A, T *B,
                    float *C, int bmmshape, MMACTFUNC func)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const float *maxc = C + M * N * bmmshape;
  const float *maxA = A + M * K * bmmshape;
  const T *OB = B;

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

  // Move C_ptr to warp's output tile

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step

  const uint innerRowA = threadIdx.x / (BK4);
  const uint innerColA = threadIdx.x % (BK4);
  const uint innerRowB = threadIdx.x / (BN4);
  const uint innerColB = threadIdx.x % (BN4);

  // allocate thread-local cache for results in registerfile

  // outer-most loop over block tiles

  float4 threadResults4[WMITER * TM * WNITER * TN4];
  auto AO = A;
  for (uint BMMINDEX = 0; BMMINDEX < bmmshape; BMMINDEX += 1)
  {
    for (uint yy = 0; yy < WMITER * TM * WNITER * TN4; yy++)
    {
      threadResults4[yy] = make_float4(0, 0, 0, 0);
    }
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float4 regN[WNITER * TN4];

    auto A = AO + cRow * BM * K * bmmshape + BMMINDEX * K;
    auto B = OB + cCol * BN + N * K * BMMINDEX;
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
      wt::loadFromGmem(
          N, K, A, maxA, B, OB + N * K * BMMINDEX, As, Bs, innerRowA, innerColA, innerRowB, innerColB, bmmshape);
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
          for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1)
          {

            auto iix = C + BMMINDEX * M * N + (cRow * BM + warpRow * WM + threadRowInWarp * TM + wSubRowIdx * WSUBM + resIdxM) * N + cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;

            if (iix < maxc)
            {
              auto zz1 = *(((float *)threadResults4) + (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN);

              auto spot = (iix);

              // iixh->x += iiy->x;
              // iixh->y += iiy->y;
              // iixh->z += iiy->z;
              // iixh->w += iiy->w;

              if (func == TANH)
              {
                spot[0] = tanh(spot[0] + zz1);
              }
              if (func == RELUSQUARE)
              {
                spot[0] += zz1;
                if (spot[0] > 0)
                {
                  spot[0] = spot[0] * spot[0];
                }
                else
                {
                  spot[0] = 0;
                }
              }
              if (func == SWISHMUL)
              {
                spot[0] = (spot[0] * zz1) / (1.0 + exp(-zz1));
              }
              if (func == SIGMOIDMUL)
              {
                spot[0] = (*(iix - C + AO - N * M)) / (1.0 + exp(-zz1)) + spot[0];
              }
              if (func == NONE)
              {
                spot[0] += zz1;
              }
              if (func == EXPNEGEXP)
              {
                spot[0] = exp(-exp(double(zz1 + spot[0])));
              }
              if (func == SETVALUE)
              {
                spot[0] = zz1;
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void runSgemmWarptiling(int M, int N, int K, float *A, T *B, float *C, int bmmshape, MMACTFUNC func)
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
  sgemmWarptiling<<<gridDim, blockDim>>>(M, N, K, A, B, C, bmmshape, func);

  cudaDeviceSynchronize();
  // get error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel failed");
  }
}

#define THREADCC 1024
#define warps 32
#define tpw 32
#define procs 128

template <int INSS>
__global__ void kernelc_mmbf16_one(
    const unsigned long long INPUTSIZE,
    const unsigned long long OUTPUTSIZE,
    __nv_bfloat162 *x,
    __nv_bfloat16 *w,
    float *y, MMACTFUNC func)
{

  auto threadid = threadIdx.x;
  auto warpid = threadid / warps;
  auto warppos = threadid % warps;

  auto outputstart = blockIdx.x * INSS;
  auto XO = x;
  auto oy = y;
  y += outputstart;
  w += INPUTSIZE * outputstart;

// for (size_t instart = 0; instart < INPUTSIZE; instart += 1024)
//   {
//     if (threadid + instart < INPUTSIZE)
//     {
//       x[(threadid + instart)].x = x[(threadid + instart)].y;
//     }
//   }
//   __syncthreads();

// atomicAdd(&ystart[warp].x, outt.x);
// __shared__ __nv_bfloat162 pool[32];
// atomicAdd(&ystart[warp].y, outt.y);
#pragma unroll
  for (size_t start = 0; start < INSS; start += 32)
  {

    float2 acc2 = make_float2(0.f,0.f);
    __nv_bfloat162 buff;
    if (start + warpid < INSS)
    {

      for (size_t instart = 0; instart < INPUTSIZE; instart += 64)
      {
        buff.x = x[(warppos * 2 + instart)].y;
        buff.y = x[(warppos * 2 + 1 + instart)].y;
        auto mxx = __bfloat1622float2(buff * *(__nv_bfloat162*)(w + warppos * 2 + instart + warpid * INPUTSIZE));
        acc2.x += mxx.x;
        acc2.y += mxx.y; 
      }
    }

    auto acc= acc2.x + acc2.y;
    for (int ij = 1; ij < warpSize; ij *= 2)
      acc += __shfl_xor_sync(-1, acc, ij);

    // pool[warpid] = acc;

    // __syncthreads();

    // acc = pool[warppos];

    // for (int ij = 1; ij < warpSize; ij *= 2)
    //   acc += __shfl_xor_sync(-1, acc, ij);
    if (warppos == 0 && start + warpid < INSS)
    {
      float zz1 = acc;

      auto spot = y + warpid;

      if (func == TANH)
      {
        spot[0] = tanh(spot[0] + zz1);
      }
      if (func == RELUSQUARE)
      {
        spot[0] += zz1;
        if (spot[0] > 0)
        {
          spot[0] = spot[0] * spot[0];
        }
        else
        {
          spot[0] = 0;
        }
      }
      if (func == SWISHMUL)
      {
        spot[0] = (spot[0] * zz1) / (1.0 + exp(-zz1));
      }
      if (func == SIGMOIDMUL)
      {
        spot[0] = (*((spot - oy) + ((float *)XO) + (-OUTPUTSIZE))) / (1.0 + exp(-zz1)) + spot[0];
      }
      if (func == NONE)
      {
        spot[0] += zz1;
      }
      if (func == EXPNEGEXP)
      {
        spot[0] = exp(-exp(double(zz1 + spot[0])));
      }
      if (func == SETVALUE)
      {
        spot[0] = zz1;
      }
    }
    w += INPUTSIZE * 32;
    y += 32;
  }

  // }
}

#define KERNELBF(O)                                                                    \
  else if (OUTSHAPE == O)                                                              \
  {                                                                                    \
    kernelc_mmbf16_one<O / procs><<<gridSize, std::min((O / procs) * 32, 1024)>>>(     \
        INSHAPE, OUTSHAPE, (__nv_bfloat162 *)B, (__nv_bfloat16 *)A, (float *)C, func); \
  }

void matmul_cuda_kernal(void *A, void *B, void *C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype, size_t bmmshape, MMACTFUNC func)
{

  dim3 gridSize(procs, 1, 1);
  // assert(tsplit%OUTSHAPE == 0);
  dim3 blockSize(1024, 1, 1);

  if (OUTSHAPE == 0 || INSHAPE == 0)
  {
    return;
  }

  if (dtype == kBFLOAT_16)
  {
    if (BBT != 1)
    {
      runSgemmWarptiling(BBT, OUTSHAPE, INSHAPE, (float *)B, (__nv_bfloat16 *)A, (float *)C, bmmshape, func);
    }
    KERNELBF(65536)
    KERNELBF(32768)
    KERNELBF(16384)
    KERNELBF(8192)
    KERNELBF(7168)
    KERNELBF(4096)
    KERNELBF(2048)
    KERNELBF(1024)
    // KERNEL(512)
    // KERNEL(256)
    // KERNEL(128)
    // KERNEL(64)
    // KERNEL(32)
    else
    {
      runSgemmWarptiling(BBT, OUTSHAPE, INSHAPE, (float *)B, (__nv_bfloat16 *)A, (float *)C, bmmshape, func);
    }
  }
  else
  {
    runSgemmWarptiling(BBT, OUTSHAPE, INSHAPE, (float *)B, (float *)A, (float *)C, bmmshape, func);
  }

  // max size 1024
}
