#ifndef SWISHMUL_NAIVE_H
#define SWISHMUL_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"


void swishmul_cpu_kernel(void* input, void* other, void* output, int size, TENSORTYPE dtype){

    size_t simdwidth = get_simd_width();

    if (dtype == TENSORTYPE::kFLOAT_32){    
        for (size_t i = 0; i < size; i+=simdwidth){
            simd_swishmul(flp(input) + i, flp(other) + i, flp(output) + i);
        }
    }
    else if (dtype == TENSORTYPE::kBFLOAT_16){ 
        for (size_t i = 0; i < size; i+=simdwidth*2){
            simd_swishmul_bf16(bflp(input) + i, bflp(other) + i, bflp(output) + i);
        }
    }
    else{
        throw std::runtime_error("swishmul only implemented for float and bfloat16");
    }
}

#endif //SWISHMUL_NAIVE_H