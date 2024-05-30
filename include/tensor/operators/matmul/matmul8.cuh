#ifndef __MATMULFP_H__
#define __MATMULFP_H__


#include "tensor/operators/matmul/kernels/matmulf.cuh"
#include "tensor/operators/matmul/kernels/matmul8.cuh"
#include "tensor/operators/matmul/kernels/wkv5.cuh"






void matmul_cuda_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE,TENSORTYPE dtype){
     

    
    if (dtype == TENSORTYPE::kFLOAT_32)
        // matmulfp_kernal<<<dimGrid, dimBlock>>>((float *)A, (float *)B, (float *)C, INSHAPE, OUTSHAPE);
        runSgemmWarptiling(BBT,INSHAPE,OUTSHAPE, (float *)A, (float *)B, (float *)C);
    else
        throw std::runtime_error("matmul not implemented for this dtype");
}








#endif