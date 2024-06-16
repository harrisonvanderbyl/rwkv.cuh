#ifndef SIMD_AVX512_H
#define SIMD_AVX512_H
#include "tensor/intrinsics/shared.h"
#include "stdint.h"

#include <cmath>

static size_t get_simd_width()
{
    return 1;
}


float __attribute__((weak))simdexpfallback(float xx)
{
    return expf(-xx);
}


void __attribute__((weak)) simd_sigmoidmul(float *input, float *other, float *residual, float *output)
{
    (*output = ((*(other)/ ((1.0f) + simdexpfallback((*(input))))) + *(residual)));
}

float __attribute__((weak)) reduce_float(float xx)
{
    auto x = flp(&xx);
    return x[0];
}

void __attribute__((weak)) simd_swishmul(float *input, float *other, float *output)
{
    (*output = ((*(float *)other * *(float *)input) / ((1.0f) + simdexpfallback(*(float *)input))));
}

void __attribute__((weak)) simd_relusquare(float *input, float *output)
{
    (*output = (*(input) * fmaxf(*(input), 0.0f)));
}

float __attribute__((weak)) simd_accumulate(float *input)
{
    return reduce_float(*(input));
}

float __attribute__((weak)) simd_variance_acc(float *input, float mean)
{
    auto v1 = *(input);
    float v2 = (v1 - (mean));
    float v3 = (v2 * v2);
    return reduce_float(v3);
}

void __attribute__((weak)) simd_lerp(float *input, float *other, float *weight, float *output)
{
    (*output = ((*(input)* ((1.0f) - *(weight))) + (*(other) * *(weight))));
}

void __attribute__((weak)) simd_norm_assign(float *input, float mean, float vareps, float *weight, float *bias, float *output)
{
    (*output = ((((*(input)- (mean))/ (vareps)) * *(weight)) + *(bias)));
}

void __attribute__((weak)) simd_tanh(float* input){
    auto x = *(input);
    auto ax = exp(x);
    auto bx = exp(-(x));
    (*input = ((ax-bx)/(ax+bx)));
}

float __attribute__((weak)) dot_uint8_floats(uint8_t *input, float *other, size_t size)
{
    auto zz1 = 0.0;

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = (*(input + k)* *(other + k)+ zz1);
    }

    return zz1;
}

float __attribute__((weak)) dot_floats(float *input, float *other, size_t size)
{
    auto zz1 = 0.0;

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = (*(input + k) * *(other + k) + zz1);
    }

    return (zz1);
}

void __attribute__((weak)) simd_wkv(size_t B, size_t T,size_t H,size_t Z, size_t bb,size_t tt, size_t hh, float *vv, float *ss, float *kk, float *uu, float *ww, float *rr, float *yy)
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
        auto acc = 0.0;
        for (size_t j = 0; j < Z; j+=16)
        {
            auto kv = *(k+j) * v[i];
            auto sss = *(s+i*Z+j);
            acc = ((kv * *(u+j) + sss)**(r+j)+acc);
            (*(s+i*Z+j) =  (sss*simdexpfallback(exp(*(w+j)))+kv));

        }

        y[i] = acc;
    }
}
#endif