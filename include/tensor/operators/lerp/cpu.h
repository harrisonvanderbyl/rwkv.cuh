#ifndef LERP_CPU_H
#define LERP_CPU_H

#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"

void lerp_cpu_kernel(void *w, void *A, void *B, void *output, size_t size, size_t loopsize, TENSORTYPE dtype)
{
    size_t simdwidth = get_simd_width();
    if (dtype == TENSORTYPE::kFLOAT_32)
    {
        for (size_t i = 0; i < size; i+=simdwidth)
        {
            float* weight = flp(w) + (i % loopsize);
            simd_lerp(flp(A) + i, flp(B) + i, weight, flp(output) + i);
        }
    }
    else if (dtype == TENSORTYPE::kBFLOAT_16)
    {
        for (size_t i = 0; i < size; i+=simdwidth*2)
        {
            bfloat16* weight = bflp(w) + i % loopsize;
            simd_lerp_bf16(bflp(A) + i, bflp(B) + i, weight, bflp(output) + i);
        }
    }
    else
    {
        throw std::runtime_error("Not implemented for this dtype");
    }
}


#endif // LERP_CPU_H