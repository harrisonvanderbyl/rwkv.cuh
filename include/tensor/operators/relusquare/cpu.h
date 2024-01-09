#ifndef RELUSQUARE_CPU_H
#define RELUSQUARE_CPU_H

#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"

void relusquare_cpu_kernel(void *input, void *output, size_t size, TENSORTYPE dtype)
{
    size_t simdwidth = get_simd_width();
    if (dtype == TENSORTYPE::kFLOAT_32)
    {
        for (size_t i = 0; i < size; i+=simdwidth)
        {
            simd_relusquare(flp(input) + i, flp(output) + i);
        }
    }
    else if (dtype == TENSORTYPE::kBFLOAT_16)
    {
        for (size_t i = 0; i < size; i+=simdwidth*2)
        {
            simd_relusquare_bf16(bflp(input) + i, bflp(output) + i);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported dtype for relusquare");
    }
}

#endif // RELUSQUARE_CPU_H