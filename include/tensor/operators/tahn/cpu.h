#ifndef tahn_NAIVE_H
#define tahn_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"


void tahn_cpu_kernel(void* input, void* output, size_t size, TENSORTYPE dtype){

    size_t simdwidth = get_simd_width();

    if (dtype == TENSORTYPE::kFLOAT_32){    
        for (size_t i = 0; i < size; i+=simdwidth){
            tahn(flp(input) + i, flp(output) + i);
        }
    }
    else if (dtype == TENSORTYPE::kBFLOAT_16){ 
        // for (size_t i = 0; i < size; i+=simdwidth*2){
        //     tahn_bf16(bflp(input) + i, bflp(other) + i, bflp(output) + i);
        // }
        throw std::runtime_error("tahn only implemented for float");
    }

    else{
        throw std::runtime_error("tahn only implemented for float and bfloat16");
    }
}

#endif //tahn_NAIVE_H