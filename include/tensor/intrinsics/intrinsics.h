#pragma once
//ifdef avx2
#ifdef __AVX512__
#include "tensor/intrinsics/avx512.h"
#elif defined(__AVX2__)
#include "tensor/intrinsics/avx2.h"
#elif defined(__ARM_NEON)
#include "tensor/intrinsics/arm.h"
#else
#include "tensor/intrinsics/fallback.h"
#endif

static inline const float sum_floats(float *input, size_t size)
{
    float sum = 0;
    for (size_t i = 0; i < size; i+=get_simd_width())
    {
        sum += simd_accumulate(input + i);
    }
    return sum;
}