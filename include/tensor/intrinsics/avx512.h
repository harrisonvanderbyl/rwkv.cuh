#ifndef SIMD_AVX512_H
#define SIMD_AVX512_H
#include "tensor/intrinsics/shared.h"

#if defined(_WIN32)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#include <cmath>

size_t get_simd_width()
{
    return 16;
}


float inline reduce_float(__m512 xx)
{
    auto x = flp(&xx);
    return _mm512_reduce_add_ps(xx);
}

float inline simd_accumulate(float *input)
{
    return reduce_float(_mm512_loadu_ps(input));
}

float inline simd_variance_acc(float *input, float mean)
{
    auto v1 = _mm512_loadu_ps(input);
    __m512 v2 = _mm512_sub_ps(v1, _mm512_set1_ps(mean));
    __m512 v3 = _mm512_mul_ps(v2, v2);
    return reduce_float(v3);
}

void inline simd_lerp(float *input, float *other, float *weighti, float *output)
{
    auto weight = (_mm512_loadu_ps(weighti));
    _mm512_storeu_ps(output, _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(input), _mm512_sub_ps(_mm512_set1_ps(1.0f), weight)), _mm512_mul_ps(_mm512_loadu_ps(other), weight)));
}


void inline simd_norm_assign(float *input, float mean, float vareps, float *weight, float *bias, float *output)
{
    _mm512_storeu_ps(output, _mm512_add_ps(_mm512_mul_ps(_mm512_div_ps(_mm512_sub_ps(_mm512_loadu_ps(input), _mm512_set1_ps(mean)), _mm512_set1_ps(vareps)), _mm512_loadu_ps(weight)), _mm512_loadu_ps(bias)));
}

float inline dot_uint8_floats(u_int8_t *input, float *other, size_t size)
{
    auto zz1 = _mm512_setzero_ps();

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_lddqu_si128((__m128i *)(input + k)))), _mm512_load_ps(other + k), zz1);
    }

    return _mm512_reduce_add_ps(zz1);
}

float inline dot_floats(float *input, float *other, size_t size)
{
    auto zz1 = _mm512_setzero_ps();

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = _mm512_fmadd_ps(_mm512_load_ps(input + k), _mm512_load_ps(other + k), zz1);
    }

    return _mm512_reduce_add_ps(zz1);
}

void inline simd_wkv(size_t B, size_t T,size_t H,size_t Z, size_t bb,size_t tt, size_t hh, float *vv, float *ss, float *kk, float *uu, float *ww, float *rr, float *yy)
{
    auto k = kk + bb*T*H*Z + tt*H*Z + hh*Z;
    auto v = vv + bb*T*H*Z + tt*H*Z + hh*Z;
    auto r = rr + bb*T*H*Z + tt*H*Z + hh*Z;
    auto y = yy + bb*T*H*Z + tt*H*Z + hh*Z;
    auto s = ss + bb*H*Z*Z + hh*Z*Z;
    auto u = uu + hh*Z;
    auto w = ww + bb*T*H*Z + tt*H*Z + hh*Z;

    for (size_t i = 0; i < Z; i++)
    {
        auto acc = _mm512_setzero_ps();
        for (size_t j = 0; j < Z; j+=16)
        {
            auto kv = _mm512_loadu_ps(k+j) * v[i];
            auto sss = _mm512_loadu_ps(s+i*Z+j);
            acc = _mm512_fmadd_ps(_mm512_fmadd_ps(kv,_mm512_loadu_ps(u+j) , sss),_mm512_loadu_ps(r+j),acc);
            _mm512_store_ps(s+i*Z+j, _mm512_fmadd_ps(sss,((((_mm512_loadu_ps(w+j))))),kv));

        }

        y[i] = _mm512_reduce_add_ps(acc);
    }
}
#endif