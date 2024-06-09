#ifndef SIMD_AVX2_H
#define SIMD_AVX2_H
#include "tensor/intrinsics/shared.h"

#if defined(_WIN32)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#include <cmath>

size_t get_simd_width()
{
    return 8;
}

#if !defined(__INTEL_LLVM_COMPILER)

__m256 simdexp256(__m256 xx)
{
    auto x = flp(&xx);
    return _mm256_set_ps(exp(-x[7]), exp(-x[6]), exp(-x[5]), exp(-x[4]), exp(-x[3]), exp(-x[2]), exp(-x[1]), exp(-x[0]));
}

#else

__m256 simdexp256(__m256 xx)
{
    return _mm256_exp_ps(-xx);
}
#endif

void inline  simd_sigmoidmul(float *input, float *other, float *residual, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_div_ps(_mm256_loadu_ps(other), _mm256_add_ps(_mm256_set1_ps(1.0f), simdexp256((_mm256_loadu_ps(input))))), _mm256_loadu_ps(residual)));
}

float inline  reduce_float(__m256 xx)
{
    auto x = flp(&xx);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
}

void inline  simd_swishmul(float *input, float *other, float *output)
{
    _mm256_storeu_ps(output, _mm256_div_ps(_mm256_mul_ps(*(__m256 *)other, *(__m256 *)input), _mm256_add_ps(_mm256_set1_ps(1.0f), simdexp256(*(__m256 *)input))));
}

void inline  simd_relusquare(float *input, float *output)
{
    _mm256_storeu_ps(output, _mm256_mul_ps(_mm256_loadu_ps(input), _mm256_max_ps(_mm256_loadu_ps(input), _mm256_setzero_ps())));
}

float inline  simd_accumulate(float *input)
{
    return reduce_float(_mm256_loadu_ps(input));
}

float inline  simd_variance_acc(float *input, float mean)
{
    auto v1 = _mm256_loadu_ps(input);
    __m256 v2 = _mm256_sub_ps(v1, _mm256_set1_ps(mean));
    __m256 v3 = _mm256_mul_ps(v2, v2);
    return reduce_float(v3);
}

void inline  simd_lerp(float *input, float *other, float *weight, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(input), _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(weight))), _mm256_mul_ps(_mm256_loadu_ps(other), _mm256_loadu_ps(weight))));
}

void inline  simd_norm_assign(float *input, float mean, float vareps, float *weight, float *bias, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(input), _mm256_set1_ps(mean)), _mm256_set1_ps(vareps)), _mm256_loadu_ps(weight)), _mm256_loadu_ps(bias)));
}

float inline  dot_uint8_floats(uint8_t *input, float *other, size_t size)
{
    auto zz1 = _mm256_setzero_ps();

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_lddqu_si128((__m128i *)(input + k)))), _mm256_load_ps(other + k), zz1);
    }

    return reduce_float(zz1);
}

float inline  dot_floats(float *input, float *other, size_t size)
{
    auto zz1 = _mm256_setzero_ps();

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = _mm256_fmadd_ps(_mm256_load_ps(input + k), _mm256_load_ps(other + k), zz1);
    }

    return reduce_float(zz1);
}

void inline  simd_wkv(size_t B, size_t T,size_t H,size_t Z, size_t bb,size_t tt, size_t hh, float *vv, float *ss, float *kk, float *uu, float *ww, float *rr, float *yy)
{
    auto k = kk + bb*T*H*Z + tt*H*Z + hh*Z;
    auto v = vv + bb*T*H*Z + tt*H*Z + hh*Z;
    auto r = rr + bb*T*H*Z + tt*H*Z + hh*Z;
    auto y = yy + bb*T*H*Z + tt*H*Z + hh*Z;
    auto s = ss + bb*H*Z*Z + hh*Z*Z;
    auto u = uu + hh*Z;
    auto w = ww + hh*Z;

    for (size_t i = 0; i < Z; i++)
    {
        auto acc = _mm256_setzero_ps();
        for (size_t j = 0; j < Z; j+=8)
        {
            auto kv = _mm256_mul_ps(_mm256_loadu_ps(k+j) , _mm256_set1_ps(v[i]));
            auto sss = _mm256_loadu_ps(s+i*Z+j);
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(kv,_mm256_loadu_ps(u+j) , sss),_mm256_loadu_ps(r+j),acc);
            _mm256_store_ps(s+i*Z+j, _mm256_fmadd_ps(sss,_mm256_loadu_ps(w+j),kv));

        }

        y[i] = reduce_float(acc);
    }
}
#endif