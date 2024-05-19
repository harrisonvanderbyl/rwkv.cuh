#ifndef MATMULF_CUH
#define MATMULF_CUH
#include "tensor/operators/matmul/kernels/globals.cuh"

__global__ void matmulfp_kernal(float*  A, float* B, float* C, size_t INSHAPE, size_t OUTSHAPE){
    size_t bbt = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ii = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    for (size_t i = ii*BLOCK_SIZE ; i < ii*BLOCK_SIZE+BLOCK_SIZE; i++)
    {
        

        float acc = float(0);

        for (size_t j = 0; j < LOOPSIZE; j++)
        {
            acc += float(A[i * INSHAPE + k*LOOPSIZE + j] * B[bbt * INSHAPE + k*LOOPSIZE + j]);
        }


        atomicAdd(C + bbt * OUTSHAPE + i, (acc));

    }
}

#endif // MATMULF_CUH