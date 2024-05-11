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

static inline const void simd_sigmoidmul(float *input, float *other, float *residual, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_div_ps(_mm256_loadu_ps(other), _mm256_add_ps(_mm256_set1_ps(1.0f), simdexp256((_mm256_loadu_ps(input))))), _mm256_loadu_ps(residual)));
}

static inline const float reduce_float(__m256 xx)
{
    auto x = flp(&xx);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
}

static inline const void simd_swishmul(float *input, float *other, float *output)
{
    _mm256_storeu_ps(output, _mm256_div_ps(_mm256_mul_ps(*(__m256 *)other, *(__m256 *)input), _mm256_add_ps(_mm256_set1_ps(1.0f), simdexp256(*(__m256 *)input))));
}

static inline const void simd_relusquare(float *input, float *output)
{
    _mm256_storeu_ps(output, _mm256_mul_ps(_mm256_loadu_ps(input), _mm256_max_ps(_mm256_loadu_ps(input), _mm256_setzero_ps())));
}

static inline const float simd_accumulate(float *input)
{
    return reduce_float(_mm256_loadu_ps(input));
}

static inline const float simd_variance_acc(float *input, float mean)
{
    auto v1 = _mm256_loadu_ps(input);
    __m256 v2 = _mm256_sub_ps(v1, _mm256_set1_ps(mean));
    __m256 v3 = _mm256_mul_ps(v2, v2);
    return reduce_float(v3);
}

static inline const void simd_lerp(float *input, float *other, float *weight, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(input), _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(weight))), _mm256_mul_ps(_mm256_loadu_ps(other), _mm256_loadu_ps(weight))));
}

static inline const void simd_norm_assign(float *input, float mean, float vareps, float *weight, float *bias, float *output)
{
    _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(input), _mm256_set1_ps(mean)), _mm256_set1_ps(vareps)), _mm256_loadu_ps(weight)), _mm256_loadu_ps(bias)));
}

static inline const float dot_uint8_floats(u_int8_t *input, float *other, size_t size)
{
    auto zz1 = _mm256_setzero_ps();

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_lddqu_si128((__m128i *)(input + k)))), _mm256_load_ps(other + k), zz1);
    }

    return reduce_float(zz1);
}

static inline const void simd_wkv(size_t size, size_t bhofseti, size_t bbhhofseti, void* vv, void* ss, float kkk, float uuu, float www, float rrr, void* out)
{
    auto rrrn = _mm256_set1_ps(rrr);
    auto wwwn = _mm256_set1_ps(www);
    for (size_t j = 0; j < size; j += get_simd_width())
    {
        size_t jind = bhofseti + j;
        size_t sind = bbhhofseti + j;

        // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
        // auto vvv = flp(vv)[jind];
        auto vvv = _mm256_loadu_ps(flp(vv)+jind);

        auto sss = _mm256_loadu_ps(flp(ss)+sind);

        // multiply kkk and vvv
        auto atu = vvv * kkk;

        // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
        auto sssatuuuu = atu * uuu + sss;

        // flp(out)[jind] += outf;
        _mm256_storeu_ps(flp(out)+jind, _mm256_fmadd_ps(sssatuuuu,rrrn,_mm256_load_ps(flp(out)+jind)));

        // *(flp(ss) + sind) = sss * www + atu;
        _mm256_storeu_ps(flp(ss)+sind, _mm256_fmadd_ps(sss,wwwn,atu));
    }
}
#endif