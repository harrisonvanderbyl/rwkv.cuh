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



void matmul8_cuda_kernal(u_char* A, void* B, void* C, void* Ao, void* Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE){  
   
   
    
    dim3 blockSize(INSHAPE/MM8_ONE_JSPLIT);
    dim3 gridSize(OUTSHAPE/MM8_ONE_TILE);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        INSHAPE, OUTSHAPE, (float*)B, A, (float*)Ar, (float*)Ao, (float*)C, BBT);
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