#ifndef SIGMOIDMUL_NAIVE_H
#define SIGMOIDMUL_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"


void sigmoidmul_cpu_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype){

    size_t simdwidth = get_simd_width();

    if (dtype == TENSORTYPE::kFLOAT_32){    
        for (size_t i = 0; i < size; i+=simdwidth){
            simd_sigmoidmul(flp(input) + i, flp(other) + i, flp(residual) + i, flp(output) + i);
        }
    }
    else if (dtype == TENSORTYPE::kBFLOAT_16){ 
        for (size_t i = 0; i < size; i+=simdwidth*2){
            simd_sigmoidmul_bf16(bflp(input) + i, bflp(other) + i, bflp(residual) + i, bflp(output) + i);
        }
    }
    else{
        throw std::runtime_error("sigmoidmul only implemented for float and bfloat16");
    }
}

#endif //SIGMOIDMUL_NAIVE_H