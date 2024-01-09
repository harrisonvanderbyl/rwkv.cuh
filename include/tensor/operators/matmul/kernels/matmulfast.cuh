#pragma once

__global__ void gemv_quantized_int8(u_int8_t* mat, float* vec, float* res,
                                    unsigned size_t n, float* scale, float* zero_point,
                                    unsigned size_t num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned size_t tid = threadIdx.x;
  unsigned size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned size_t start_idx = threadIdx.x;
  double4* mat4 = reinterpret_cast<double4*>(mat);
  double4* vec4 = reinterpret_cast<double4*>(vec);


#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned size_t j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      double4 vec_val = vec4[j]; // actually float4x2
      double4 mat_val = mat4[row * (n >> 3) + j];
      const float2* vec_h1 = (float2*)&vec_val.x;
      const float2* vec_h2 = (float2*)&vec_val.y;
      const float2* vec_h3 = (float2*)&vec_val.z;
      const float2* vec_h4 = (float2*)&vec_val.w;
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) *
             (static_cast<float>(mat_h1->x) + zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
             (static_cast<float>(mat_h1->y) + zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
             (static_cast<float>(mat_h2->x) + zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
             (static_cast<float>(mat_h2->y) + zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
             (static_cast<float>(mat_h3->x) + zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
             (static_cast<float>(mat_h3->y) + zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
             (static_cast<float>(mat_h4->x) + zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
             (static_cast<float>(mat_h4->y) + zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2float(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const size_t laneId = threadIdx.x % WARP_SIZE;
  const size_t warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2float(sum);
  }
}

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned size_t threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}