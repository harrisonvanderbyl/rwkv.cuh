#if !defined(TENSOR_INTRINSICS_ARM_H)
#define TENSOR_INTRINSICS_ARM_H
#include "tensor/intrinsics/shared.h"
#include <arm_neon.h>

#if !defined(vdivq_f32)
#define vdivq_f32(x, y) vdivcompatq_f32(x, y)
float32x4_t vdivcompatq_f32(float32x4_t x, float32x4_t y)
{
    float32x4_t recip = vrecpeq_f32(y);
    recip = vmulq_f32(vrecpsq_f32(y, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(y, recip), recip);
    return vmulq_f32(x, recip);
}
#endif

size_t get_simd_width()
{
    return 4;
}

float32x4_t arm_exp(float32x4_t x)
{
    float32x4_t xx = vdupq_n_f32(0);
    xx[0] = exp(x[0]);
    xx[1] = exp(x[1]);
    xx[2] = exp(x[2]);
    xx[3] = exp(x[3]);
    return xx;
}

    void simd_sigmoidmul(float *input, float *other, float *residual, float *output)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vld1q_f32(other);
    float32x4_t v3 = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1)));
    // return v2 / v3;
    vst1q_f32(output, vaddq_f32(vdivq_f32(v2, v3), vld1q_f32(residual)));
}

void simd_swishmul(float *input, float *other, float *output)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vld1q_f32(other);
    float32x4_t v3 = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1)));

    // return v2 * v1 / v3;
    vst1q_f32(output, vmulq_f32(v2, vdivq_f32(v1, v3)));
}

void simd_relusquare(float *input, float *output)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vmaxq_f32(v1, vdupq_n_f32(0.0f));
    vst1q_f32(output, vmulq_f32(v1, v2));
}

float simd_accumulate(float *input)
{
    // arm neon
    return input[0] + input[1] + input[2] + input[3];
}

float reduce_float(float32x4_t xx)
{
    return xx[0] + xx[1] + xx[2] + xx[3];
}

    float simd_variance_acc(float *input, float mean)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vsubq_f32(v1, vdupq_n_f32(mean));
    float32x4_t v3 = vmulq_f32(v2, v2);
    return v3[0] + v3[1] + v3[2] + v3[3];
}

void simd_lerp(float *input, float *other, float *weight, float *output)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vsubq_f32(vdupq_n_f32(1.0f), vld1q_f32(weight));
    float32x4_t v3 = vmulq_f32(v1, v2);
    float32x4_t v4 = vmulq_f32(vld1q_f32(other), vld1q_f32(weight));
    float32x4_t v5 = vaddq_f32(v3, v4);
    vst1q_f32(output, v5);
}
void simd_norm_assign(float *input, float mean, float vareps, float *weight, float *bias, float *output)
{
    // arm neon
    float32x4_t v1 = vld1q_f32(input);
    float32x4_t v2 = vsubq_f32(v1, vdupq_n_f32(mean));
    float32x4_t v3 = vdivq_f32(v2, vdupq_n_f32(vareps));
    float32x4_t v4 = vmulq_f32(v3, vld1q_f32(weight));
    float32x4_t v5 = vaddq_f32(v4, vld1q_f32(bias));
    vst1q_f32(output, v5);
}



static inline const float dot_uint8_floats(u_int8_t *input, float *other, size_t size)
{
    auto zz1 = vdupq_n_f32(0);

    for (auto k = 0; k < size; k+=get_simd_width()*2)
    {
        auto u16vec = vmovl_u8(vld1_u8(input + k));
        auto u32vec = vmovl_u16(vget_low_u16(u16vec));
        auto f32vec = vcvtq_f32_u32(u32vec);
        zz1 = vfmaq_f32(zz1, f32vec, vld1q_f32(other + k));

        u32vec = vmovl_u16(vget_high_u16(u16vec));
        f32vec = vcvtq_f32_u32(u32vec);

        zz1 = vfmaq_f32(zz1, f32vec, vld1q_f32(other + k + get_simd_width()));
;
    }

    return reduce_float(zz1);
}


static inline const float dot_floats(float *input, float *other, size_t size)
{
    auto zz1 = vdupq_n_f32(0);

    for (auto k = 0; k < size; k += get_simd_width())
    {
        zz1 = vfmaq_f32(zz1, vld1q_f32(input + k), vld1q_f32(other + k));
    }

    return reduce_float(zz1);
}

static inline const void simd_wkv(size_t B, size_t T,size_t H,size_t Z, size_t bb,size_t tt, size_t hh, float *vv, float *ss, float *kk, float *uu, float *ww, float *rr, float *yy)
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
        auto acc = vdupq_n_f32(0);
        for (size_t j = 0; j < Z; j+=get_simd_width())
        {
            auto kv = vmulq_f32(vld1q_f32(k+j), vdupq_n_f32(v[i]));
            auto sss = vld1q_f32(s+i*Z+j);
            acc = vfmaq_f32(acc, vfmaq_f32(sss,vld1q_f32(u+j) , kv),vld1q_f32(r+j));
            vst1q_f32(s+i*Z+j, vfmaq_f32(kv,vld1q_f32(w+j),sss));

        }

        y[i] = reduce_float(acc);
    }
}


#endif