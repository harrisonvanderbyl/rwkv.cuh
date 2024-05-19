#ifndef __MATMULFP_H__
#define __MATMULFP_H__


#include "tensor/operators/matmul/kernels/matmulf.cuh"
#include "tensor/operators/matmul/kernels/matmul8.cuh"
#include "tensor/operators/matmul/kernels/wkv5.cuh"




size_t VSPLIT = 16;
size_t HSPLIT = 8;



void matmul_cuda_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE,TENSORTYPE dtype){
     

    
    dim3 dimBlock(1, VSPLIT, HSPLIT);
    dim3 dimGrid(BBT, ((OUTSHAPE/VSPLIT)/BLOCK_SIZE), ((INSHAPE/HSPLIT)/LOOPSIZE));
    if (dtype == TENSORTYPE::kFLOAT_32)
        // matmulfp_kernal<<<dimGrid, dimBlock>>>((float *)A, (float *)B, (float *)C, INSHAPE, OUTSHAPE);
        matmulfp_kernal<<<dimGrid, dimBlock>>>((float *)A, (float *)B, (float *)C, INSHAPE, OUTSHAPE);
    else if (dtype == TENSORTYPE::kBFLOAT_16)
        matmulfp_kernal<<<dimGrid, dimBlock>>>((__nv_bfloat16 *)A, (__nv_bfloat16 *)B, (__nv_bfloat16 *)C, INSHAPE, OUTSHAPE);
    else
        throw std::runtime_error("matmul not implemented for this dtype");
}








#endif