#ifndef GLOBALCUDA_H
#define GLOBALCUDA_H
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tensor/tensor.h"

#define BLOCK_SIZE 1
#define LOOPSIZE 32

// #define MM8_ONE_JTILE 8
#define MM8_ONE_JSPLIT 32
#define MM8_ONE_TILE 32


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const static int WARPSIZE = 32; // warpSize is not constexpr
// #define BLOCKSPLIT 8

__device__ struct bfloat1624
{
  __nv_bfloat162 x = __float2bfloat162_rn(0.0f);
  __nv_bfloat162 y = __float2bfloat162_rn(0.0f);
  __device__ bfloat1624(float4 in){
    x = __float22bfloat162_rn(*(float2*)&in.x);
    y = __float22bfloat162_rn(*(float2*)&in.z);
  };
  __device__ bfloat1624(float a, float b, float c, float d){
    x = __floats2bfloat162_rn(a,b);
    y = __floats2bfloat162_rn(c,d);
  }
  __device__ void fma(bfloat1624 r, bfloat1624 o){
    x = __hfma2(x,r.x,o.x);
    y = __hfma2(y,r.y,o.y);
  }
};


#endif // GLOBALCUDA_H