#pragma once
//ifdef avx2
#ifdef __CUDACC__
#include "tensor/intrinsics/fallback.h"
#pragma message "Using fallback, you should not include the cpu operators from your cuda code"
#else

#ifdef __AVX512__
#include "tensor/intrinsics/avx512.h"
#pragma message "Using AVX512"
#elif defined(__AVX2__)
#include "tensor/intrinsics/avx2.h"
#pragma message "Using AVX2"
#elif defined(__ARM_NEON)
#include "tensor/intrinsics/arm.h"
#pragma message "Using ARM"
#else
#include "tensor/intrinsics/fallback.h"
#pragma message "No simd intrinsics found, using fallback"
#endif
#endif

float static inline sum_floats(float *input, size_t size)
{
    float sum = 0;
    for (size_t i = 0; i < size; i+=get_simd_width())
    {
        sum += simd_accumulate(input + i);
    }
    return sum;
}