#ifndef INTRINSICS_H
#define INTRINSICS_H

#if defined(__ARM_NEON)
#include <arm_neon.h>
#include <arm_bf16.h>
#else
#include <immintrin.h>
#endif
#include <iostream>


#define flp(x) ((float*)(x))
#define bflp(x) ((bfloat16*)(x))

#if defined(__ARM_NEON)
#define ARMONLY(x) x
#define AVXONLY(x)
#else
#define ARMONLY(x) 
#define AVXONLY(x) x
#endif


// simd width
AVXONLY(
__attribute__ ((target ("avx512f")))
size_t get_simd_width(){
    return 16;
}

__attribute__ ((target ("avx2")))
size_t get_simd_width(){
    return 8;
}

__attribute__ ((target ("default")))
size_t get_simd_width(){
    return 1;
}
)

ARMONLY(
    size_t get_simd_width(){
        return 4;
    }
)

// if using intel compiler, use _mm512_exp_ps
// if using gcc, use expf
#if !defined(__INTEL_LLVM_COMPILER)
AVXONLY(
__attribute__ ((target ("avx512er")))
__m512 _mm512_exp_ps(__m512 x){
    return _mm512_exp_ps(x);
}

__attribute__ ((target ("avx512f")))
__m512 _mm512_exp_ps(__m512 x){
    return _mm512_set_ps(exp(x[15]), exp(x[14]), exp(x[13]), exp(x[12]), exp(x[11]), exp(x[10]), exp(x[9]), exp(x[8]), exp(x[7]), exp(x[6]), exp(x[5]), exp(x[4]), exp(x[3]), exp(x[2]), exp(x[1]), exp(x[0]));
};

__attribute__ ((target ("avx2")))
__m256 _mm256_exp_ps(__m256 x){
    return _mm256_set_ps(exp(x[7]), exp(x[6]), exp(x[5]), exp(x[4]), exp(x[3]), exp(x[2]), exp(x[1]), exp(x[0]));
}


)
#endif



// simd exp for arm
ARMONLY(
    float32x4_t arm_exp(float32x4_t x){
        float32x4_t xx = vdupq_n_f32(0);
        xx[0] = exp(x[0]);
        xx[1] = exp(x[1]);
        xx[2] = exp(x[2]);
        xx[3] = exp(x[3]);
        return xx;
    }
)




// simd sigmoidmul
AVXONLY(
    __attribute__ ((target ("avx2")))
    float reduce_float(__m256 x){
        return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
    }

    __attribute__ ((target ("avx512f")))
    void simd_sigmoidmul(float* input, float* other, float* residual, float* output){
        _mm512_storeu_ps(output, _mm512_add_ps(_mm512_div_ps(_mm512_loadu_ps(other), _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-_mm512_loadu_ps(input)))), _mm512_loadu_ps(residual)));
    }

    __attribute__ ((target ("avx2")))
    void simd_sigmoidmul(float* input, float* other, float* residual, float* output){
        _mm256_storeu_ps(output, _mm256_add_ps(_mm256_div_ps(_mm256_loadu_ps(other), _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-_mm256_loadu_ps(input)))), _mm256_loadu_ps(residual)));
    }

    __attribute__ ((target ("default")))
    void simd_sigmoidmul(float* input, float* other, float* residual, float* output){
        *output = *other / (1.0f + exp(-*input)) + *residual;
    }
)

ARMONLY(
    void simd_sigmoidmul(float* input, float* other, float* residual, float* output){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vld1q_f32(other);
        float32x4_t v3 = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1)));
        // return v2 / v3;
        vst1q_f32(output, vaddq_f32(vdivq_f32(v2, v3), vld1q_f32(residual)));
    }
)

// bfloat16

// ((bfloat16*)(output))[i] = bfloat16(float(((bfloat16*)(other))[i]) / (1.0f + exp(-float(((bfloat16*)(input))[i]))));

AVXONLY(

    __attribute__ ((target ("avx512f")))
    __m512 bf16_float32(__m256i x){
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
    }

    __attribute__ ((target ("avx512bf16")))
    __m512 bf16_float32(__m256i x){
        return _mm512_cvtpbh_ps((__m256bh)x);
        
    }

    __attribute__ ((target ("avx512f")))
    __m512i float32_bf16(__m512 y, __m512 x){
        return _mm512_inserti64x4(_mm512_inserti64x4(_mm512_setzero_epi32(), _mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512(x),16)), 0), _mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512(y),16)), 1);
    }

    __attribute__ ((target ("avx512bf16")))
    __m512i float32_bf16(__m512 y, __m512 x){
        return (__m512i)_mm512_cvtne2ps_pbh(y, x);   
    }

    __attribute__ ((target ("avx2")))
    __m256 bf16_float32_avx2(__m128i x){
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16));
    }

    __attribute__ ((target ("avx2")))
    __m256i float32_bf16_avx2(__m256 y, __m256 x){
        return _mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(_mm256_castps_si256(x), 16), _mm256_srai_epi32(_mm256_castps_si256(y), 16)), 0b11011000);
        // return 
    }

    
)

AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_sigmoidmul_bf16(bfloat16* input, bfloat16* other, bfloat16* residual, bfloat16* output){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));
        auto v2 = _mm512_loadu_si512(other);
        __m512 v2u = bf16_float32(_mm512_extracti64x4_epi64(v2, 0));
        __m512 v2l = bf16_float32(_mm512_extracti64x4_epi64(v2, 1));
        auto v3 = _mm512_loadu_si512(residual);
        __m512 v3u = bf16_float32(_mm512_extracti64x4_epi64(v3, 0));
        __m512 v3l = bf16_float32(_mm512_extracti64x4_epi64(v3, 1));

        __m512 v4u = _mm512_add_ps(_mm512_div_ps(v2u, _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-v1u))), v3u);
        __m512 v4l = _mm512_add_ps(_mm512_div_ps(v2l, _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-v1l))), v3l);
             
        _mm512_storeu_si512(output, float32_bf16(v4l, v4u));
    }

    __attribute__ ((target ("avx2")))
    void simd_sigmoidmul_bf16(bfloat16* input, bfloat16* other, bfloat16* residual, bfloat16* output){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));
        auto v2 = _mm256_loadu_si256((__m256i*)other);
        __m256 v2u = bf16_float32_avx2(_mm256_extracti128_si256(v2, 0));
        __m256 v2l = bf16_float32_avx2(_mm256_extracti128_si256(v2, 1));
        auto v3 = _mm256_loadu_si256((__m256i*)input);
        __m256 v3u = bf16_float32_avx2(_mm256_extracti128_si256(v3, 0));
        __m256 v3l = bf16_float32_avx2(_mm256_extracti128_si256(v3, 1));

        __m256 v4u = _mm256_add_ps(_mm256_div_ps(v2u, _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-v1u))), v3u);
        __m256 v4l = _mm256_add_ps(_mm256_div_ps(v2l, _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-v1l))), v3l);
             
        _mm256_storeu_si256((__m256i*)output, float32_bf16_avx2(v4l, v4u));
    }

    __attribute__ ((target ("default")))
    void simd_sigmoidmul_bf16(bfloat16* input, bfloat16* other, bfloat16* residual, bfloat16* output){
       
        *(output) = bfloat16(float(*(other)) / (1.0f + exp(-float(*(input)))) + float(*(residual)));
        
    }
)

ARMONLY(
    __attribute__ ((target ("bf16")))
    void simd_sigmoidmul_bf16(bfloat16* input, bfloat16* other, bfloat16* residual, bfloat16* output){
        // arm neon
        bfloat16x8_t v1 = vld1q_bf16(input);
        bfloat16x8_t v2 = vld1q_bf16(other);
        bfloat16x8_t v3 = vld1q_bf16(residual);
        
        float32x4_t v1upper = vcvt_f32_bf16(vget_high_bf16(v1));
        float32x4_t v1lower = vcvt_f32_bf16(vget_low_bf16(v1));

        float32x4_t v2upper = vcvt_f32_bf16(vget_high_bf16(v2));
        float32x4_t v2lower = vcvt_f32_bf16(vget_low_bf16(v2));

        float32x4_t v3upper = vcvt_f32_bf16(vget_high_bf16(v3));
        float32x4_t v3lower = vcvt_f32_bf16(vget_low_bf16(v3));

        float32x4_t v4upper = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1upper)));
        float32x4_t v4lower = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1lower)));

        float32x4_t v5upper = vaddq_f32(vdivq_f32(v2upper, v4upper), v3upper);
        float32x4_t v5lower = vaddq_f32(vdivq_f32(v2lower, v4lower), v3lower);
 

        vst1q_bf16(output, vcombine_bf16(vcvt_bf16_f32(v5lower), vcvt_bf16_f32(v5upper)));
        // store
    }
)

// swishmul
// simd sigmoidmul
AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_swishmul(float* input, float* other, float* output){
        _mm512_storeu_ps(output, _mm512_div_ps(_mm512_mul_ps(*(__m512*)other,*(__m512*)input), _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-*(__m512*)input))));
    }

    __attribute__ ((target ("avx2")))
    void simd_swishmul(float* input, float* other, float* output){
        _mm256_storeu_ps(output, _mm256_div_ps(_mm256_mul_ps(*(__m256*)other,*(__m256*)input), _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-*(__m256*)input))));
    }

    __attribute__ ((target ("default")))
    void simd_swishmul(float* input, float* other, float* output){
        *output = (*other * *input) / (1.0f + exp(-*input));
    }
)

ARMONLY(
    void simd_swishmul(float* input, float* other, float* output){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vld1q_f32(other);
        float32x4_t v3 = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1)));

        // return v2 * v1 / v3;
        vst1q_f32(output, vmulq_f32(v2, vdivq_f32(v1, v3)));
    }
)

AVXONLY(


    __attribute__ ((target ("avx512f")))
    void simd_swishmul_bf16(bfloat16* input, bfloat16* other, bfloat16* output){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));
        auto v2 = _mm512_loadu_si512(other);
        __m512 v2u = bf16_float32(_mm512_extracti64x4_epi64(v2, 0));
        __m512 v2l = bf16_float32(_mm512_extracti64x4_epi64(v2, 1));

        __m512 v3u = _mm512_div_ps(_mm512_mul_ps(v2u,v1u), _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-v1u)));
        __m512 v3l = _mm512_div_ps(_mm512_mul_ps(v2l,v1l), _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_exp_ps(-v1l)));
             
        _mm512_storeu_si512(output, float32_bf16(v3l, v3u));
    }

    __attribute__ ((target ("avx2")))
    void simd_swishmul_bf16(bfloat16* input, bfloat16* other, bfloat16* output){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));
        auto v2 = _mm256_loadu_si256((__m256i*)other);
        __m256 v2u = bf16_float32_avx2(_mm256_extracti128_si256(v2, 0));
        __m256 v2l = bf16_float32_avx2(_mm256_extracti128_si256(v2, 1));

        __m256 v3u = _mm256_div_ps(_mm256_mul_ps(v2u,v1u), _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-v1u)));
        __m256 v3l = _mm256_div_ps(_mm256_mul_ps(v2l,v1l), _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(-v1l)));
             
        _mm256_storeu_si256((__m256i*)output, float32_bf16_avx2(v3l, v3u));
    }

    __attribute__ ((target ("default")))
    void simd_swishmul_bf16(bfloat16* input, bfloat16* other, bfloat16* output){
       
        *(output) = bfloat16((float(*(other)) * float(*(input))) / (1.0f + exp(-float(*(input)))));
        
    }
)

ARMONLY(
    __attribute__ ((target ("bf16")))
    void simd_swishmul_bf16(bfloat16* input, bfloat16* other, bfloat16* output){
        // arm neon
        bfloat16x8_t v1 = vld1q_bf16(input);
        bfloat16x8_t v2 = vld1q_bf16(other);
        
        float32x4_t v1upper = vcvt_f32_bf16(vget_high_bf16(v1));
        float32x4_t v1lower = vcvt_f32_bf16(vget_low_bf16(v1));

        float32x4_t v2upper = vcvt_f32_bf16(vget_high_bf16(v2));
        float32x4_t v2lower = vcvt_f32_bf16(vget_low_bf16(v2));

        float32x4_t v3upper = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1upper)));
        float32x4_t v3lower = vaddq_f32(vdupq_n_f32(1.0f), arm_exp(vnegq_f32(v1lower)));

        float32x4_t v4upper = vmulq_f32(v2upper, vmulq_f32(v1upper, vrecpeq_f32(v3upper)));
        float32x4_t v4lower = vmulq_f32(v2lower, vmulq_f32(v1lower, vrecpeq_f32(v3lower)));

        vst1q_bf16(output, vcombine_bf16(vcvt_bf16_f32(v4lower), vcvt_bf16_f32(v4upper)));

        // store
    }
)

// relusquare
AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_relusquare(float* input, float* output){
        _mm512_storeu_ps(output, _mm512_mul_ps(_mm512_loadu_ps(input), _mm512_max_ps(_mm512_loadu_ps(input), _mm512_setzero_ps())));
    }

    __attribute__ ((target ("avx2")))
    void simd_relusquare(float* input, float* output){
        _mm256_storeu_ps(output, _mm256_mul_ps(_mm256_loadu_ps(input), _mm256_max_ps(_mm256_loadu_ps(input), _mm256_setzero_ps())));
    }

    __attribute__ ((target ("default")))
    void simd_relusquare(float* input, float* output){
        // max(0, x)^2
        *output = std::max(0.0f, *input) * *input;
    }
)

ARMONLY(
    void simd_relusquare(float* input, float* output){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vmaxq_f32(v1, vdupq_n_f32(0.0f));
        vst1q_f32(output, vmulq_f32(v1, v2));
    }
)

// bf16
AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_relusquare_bf16(bfloat16* input, bfloat16* output){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));

        __m512 v2u = _mm512_max_ps(v1u, _mm512_setzero_ps());
        __m512 v2l = _mm512_max_ps(v1l, _mm512_setzero_ps());

        __m512 v3u = _mm512_mul_ps(v1u, v2u);
        __m512 v3l = _mm512_mul_ps(v1l, v2l);
             
        _mm512_storeu_si512(output, float32_bf16(v3l, v3u));
    }

    __attribute__ ((target ("avx2")))
    void simd_relusquare_bf16(bfloat16* input, bfloat16* output){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));

        __m256 v2u = _mm256_max_ps(v1u, _mm256_setzero_ps());
        __m256 v2l = _mm256_max_ps(v1l, _mm256_setzero_ps());

        __m256 v3u = _mm256_mul_ps(v1u, v2u);
        __m256 v3l = _mm256_mul_ps(v1l, v2l);
             
        _mm256_storeu_si256((__m256i*)output, float32_bf16_avx2(v3l, v3u));
    }

    __attribute__ ((target ("default")))
    void simd_relusquare_bf16(bfloat16* input, bfloat16* output){
       
        *(output) = bfloat16(std::max(0.0f, float(*(input))) * float(*(input)));

    }
)

ARMONLY(
    __attribute__ ((target ("bf16")))
    void simd_relusquare_bf16(bfloat16* input, bfloat16* output){
        // arm neon
        bfloat16x8_t v1 = vld1q_bf16(input);
        float32x4_t v1upper = vcvt_f32_bf16(vget_high_bf16(v1));
        float32x4_t v1lower = vcvt_f32_bf16(vget_low_bf16(v1));

        float32x4_t v2upper = vmaxq_f32(v1upper, vdupq_n_f32(0.0f));
        float32x4_t v2lower = vmaxq_f32(v1lower, vdupq_n_f32(0.0f));

        float32x4_t v3upper = vmulq_f32(v1upper, v2upper);
        float32x4_t v3lower = vmulq_f32(v1lower, v2lower);

        vst1q_bf16(output, vcombine_bf16(vcvt_bf16_f32(v3lower), vcvt_bf16_f32(v3upper)));
        // store
    }
)

// accumulate

AVXONLY(
    __attribute__ ((target ("avx512f")))
    float simd_accumulate(float* input){
        return _mm512_reduce_add_ps(_mm512_loadu_ps(input));
    }

    __attribute__ ((target ("avx2")))
    float simd_accumulate(float* input){
        return reduce_float(_mm256_loadu_ps(input));
    }

    __attribute__ ((target ("default")))
    float simd_accumulate(float* input){
        return *input;
    }
)

ARMONLY(
    float simd_accumulate(float* input){
        // arm neon
        return input[0] + input[1] + input[2] + input[3];
    }
)

// accumulate bf16

AVXONLY(
    __attribute__ ((target ("avx512f")))
    float simd_accumulate_bf16(bfloat16* input){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));

        return _mm512_reduce_add_ps(v1u) + _mm512_reduce_add_ps(v1l);
    }

    __attribute__ ((target ("avx2")))
    float simd_accumulate_bf16(bfloat16* input){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));

        return reduce_float(v1u) + reduce_float(v1l);
    }

    __attribute__ ((target ("default")))
    float simd_accumulate_bf16(bfloat16* input){
        return float(*input);
    }
)

ARMONLY(
    __attribute__ ((target ("bf16")))
    float simd_accumulate_bf16(bfloat16* input){
        // arm neon
        auto a = vcvt_f32_bf16(vld1_bf16(input));
        auto b = vcvt_f32_bf16(vld1_bf16(input+4));

        return  a[0]+a[1]+a[2]+a[3]+b[0]+b[1]+b[2]+b[3];
    }
)

// variance
// (((float*)this->data)[j] - mean) * (((float*)this->data)[j] - mean)

AVXONLY(
    __attribute__ ((target ("avx512f")))
    float simd_variance_acc(float* input, float mean){
        auto v1 = _mm512_loadu_ps(input);
        __m512 v2 = _mm512_sub_ps(v1, _mm512_set1_ps(mean));
        __m512 v3 = _mm512_mul_ps(v2, v2);
        return _mm512_reduce_add_ps(v3);
    }

    __attribute__ ((target ("avx2")))
    float simd_variance_acc(float* input, float mean){
        auto v1 = _mm256_loadu_ps(input);
        __m256 v2 = _mm256_sub_ps(v1, _mm256_set1_ps(mean));
        __m256 v3 = _mm256_mul_ps(v2, v2);
        return reduce_float(v3);
    }

    __attribute__ ((target ("default")))
    float simd_variance_acc(float* input, float mean){
        return (*input - mean) * (*input - mean);
    }

)

ARMONLY(
    float simd_variance_acc(float* input, float mean){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vsubq_f32(v1, vdupq_n_f32(mean));
        float32x4_t v3 = vmulq_f32(v2, v2);
        return v3[0]+v3[1]+v3[2]+v3[3];
    }
)

// variance bf16
AVXONLY(
    __attribute__ ((target ("avx512f")))
    float simd_variance_acc_bf16(bfloat16* input, float mean){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));

        __m512 v2u = _mm512_sub_ps(v1u, _mm512_set1_ps(mean));
        __m512 v2l = _mm512_sub_ps(v1l, _mm512_set1_ps(mean));

        __m512 v3u = _mm512_mul_ps(v2u, v2u);
        __m512 v3l = _mm512_mul_ps(v2l, v2l);

        return _mm512_reduce_add_ps(v3u) + _mm512_reduce_add_ps(v3l);
    }

    __attribute__ ((target ("avx2")))
    float simd_variance_acc_bf16(bfloat16* input, float mean){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));

        __m256 v2u = _mm256_sub_ps(v1u, _mm256_set1_ps(mean));
        __m256 v2l = _mm256_sub_ps(v1l, _mm256_set1_ps(mean));

        __m256 v3u = _mm256_mul_ps(v2u, v2u);
        __m256 v3l = _mm256_mul_ps(v2l, v2l);

        return reduce_float(v3u) + reduce_float(v3l);
    }

    __attribute__ ((target ("default")))
    float simd_variance_acc_bf16(bfloat16* input, float mean){
        return (float(*input) - mean) * (float(*input) - mean);
    }
)

ARMONLY(
    __attribute__ ((target ("bf16")))
    float simd_variance_acc_bf16(bfloat16* input, float mean){
        // arm neon
        bfloat16x8_t v1 = vld1q_bf16(input);
        float32x4_t v1upper = vcvt_f32_bf16(vget_high_bf16(v1));
        float32x4_t v1lower = vcvt_f32_bf16(vget_low_bf16(v1));

        float32x4_t v2upper = vsubq_f32(v1upper, vdupq_n_f32(mean));
        float32x4_t v2lower = vsubq_f32(v1lower, vdupq_n_f32(mean));

        float32x4_t v3upper = vmulq_f32(v2upper, v2upper);
        float32x4_t v3lower = vmulq_f32(v2lower, v2lower);

        return v3upper[0] + v3upper[1] + v3upper[2] + v3upper[3] + v3lower[0] + v3lower[1] + v3lower[2] + v3lower[3];
    }
)


// sqrt and assign
// ((float*)result.data)[j] = ((((float*)this->data)[j] - mean) / sqrt(var + eps)) * ((float*)weight.data)[j%lastshape] + ((float*)bias.data)[j%lastshape];

AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_norm_assign(float* input, float mean, float vareps, float* weight, float* bias, float* output){
        _mm512_storeu_ps(output, _mm512_fmadd_ps(_mm512_div_ps(_mm512_sub_ps(_mm512_loadu_ps(input), _mm512_set1_ps(mean)), _mm512_set1_ps(vareps)), _mm512_loadu_ps(weight), _mm512_loadu_ps(bias)));
    }

    __attribute__ ((target ("avx2")))
    void simd_norm_assign(float* input, float mean, float vareps, float* weight, float* bias, float* output){
        _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(input), _mm256_set1_ps(mean)), _mm256_set1_ps(vareps)), _mm256_loadu_ps(weight)), _mm256_loadu_ps(bias)));
    }

    __attribute__ ((target ("default")))
    void simd_norm_assign(float* input, float mean, float vareps, float* weight, float* bias, float* output){
        *output = ((*input - mean) / vareps) * *weight + *bias;
    }

)

ARMONLY(
    void simd_norm_assign(float* input, float mean, float vareps, float* weight, float* bias, float* output){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vsubq_f32(v1, vdupq_n_f32(mean));
        float32x4_t v3 = vdivq_f32(v2, vdupq_n_f32(vareps));
        float32x4_t v4 = vmulq_f32(v3, vld1q_f32(weight));
        float32x4_t v5 = vaddq_f32(v4, vld1q_f32(bias));
        vst1q_f32(output, v5);
    }
)

// sqrt and assign bf16
AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_norm_assign_bf16(bfloat16* input, float mean, float vareps, bfloat16* weight, bfloat16* bias, bfloat16* output){
        auto v1 = _mm512_loadu_si512(input);
        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(v1, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(v1, 1));

        __m512 v2u = _mm512_sub_ps(v1u, _mm512_set1_ps(mean));
        __m512 v2l = _mm512_sub_ps(v1l, _mm512_set1_ps(mean));

        __m512 v3u = _mm512_div_ps(v2u, _mm512_set1_ps(vareps));
        __m512 v3l = _mm512_div_ps(v2l, _mm512_set1_ps(vareps));

        __m512 v4u = bf16_float32(_mm512_extracti64x4_epi64(_mm512_loadu_si512(weight), 0));
        __m512 v4l = bf16_float32(_mm512_extracti64x4_epi64(_mm512_loadu_si512(weight), 1));

        __m512 v5u = _mm512_fmadd_ps(v3u, v4u, bf16_float32(_mm512_extracti64x4_epi64(_mm512_loadu_si512(bias), 0)));
        __m512 v5l = _mm512_fmadd_ps(v3l, v4l, bf16_float32(_mm512_extracti64x4_epi64(_mm512_loadu_si512(bias), 1)));

        _mm512_storeu_si512(output, float32_bf16(v5l, v5u));
    }

    __attribute__ ((target ("avx2")))
    void simd_norm_assign_bf16(bfloat16* input, float mean, float vareps, bfloat16* weight, bfloat16* bias, bfloat16* output){
        auto v1 = _mm256_loadu_si256((__m256i*)input);
        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));

        __m256 v2u = _mm256_sub_ps(v1u, _mm256_set1_ps(mean));
        __m256 v2l = _mm256_sub_ps(v1l, _mm256_set1_ps(mean));

        __m256 v3u = _mm256_div_ps(v2u, _mm256_set1_ps(vareps));
        __m256 v3l = _mm256_div_ps(v2l, _mm256_set1_ps(vareps));

        __m256 v4u = bf16_float32_avx2(_mm256_extracti128_si256(_mm256_loadu_si256((__m256i_u*)weight), 0));
        __m256 v4l = bf16_float32_avx2(_mm256_extracti128_si256(_mm256_loadu_si256((__m256i_u*)weight), 1));

        __m256 v5u = _mm256_add_ps(_mm256_mul_ps(v3u, v4u), bf16_float32_avx2(_mm256_extracti128_si256(_mm256_loadu_si256((__m256i_u*)bias), 0)));
        __m256 v5l = _mm256_add_ps(_mm256_mul_ps(v3l, v4l), bf16_float32_avx2(_mm256_extracti128_si256(_mm256_loadu_si256((__m256i_u*)bias), 1)));

        _mm256_storeu_si256((__m256i*)output, float32_bf16_avx2(v5l, v5u));

    }

    __attribute__ ((target ("default")))
    void simd_norm_assign_bf16(bfloat16* input, float mean, float vareps, bfloat16* weight, bfloat16* bias, bfloat16* output){
        *(output) = bfloat16((float(*(input)) - mean) / vareps * float(*(weight)) + float(*(bias)));
    }

)

ARMONLY(
    __attribute__ ((target ("bf16")))
    void simd_norm_assign_bf16(bfloat16* input, float mean, float vareps, bfloat16* weight, bfloat16* bias, bfloat16* output){
        // arm neon
        bfloat16x8_t v1 = vld1q_bf16(input);
        float32x4_t v1upper = vcvt_f32_bf16(vget_high_bf16(v1));
        float32x4_t v1lower = vcvt_f32_bf16(vget_low_bf16(v1));

        float32x4_t v2upper = vsubq_f32(v1upper, vdupq_n_f32(mean));
        float32x4_t v2lower = vsubq_f32(v1lower, vdupq_n_f32(mean));

        float32x4_t v3upper = vdivq_f32(v2upper, vdupq_n_f32(vareps));
        float32x4_t v3lower = vdivq_f32(v2lower, vdupq_n_f32(vareps));

        float32x4_t v4upper = vcvt_f32_bf16(vld1_bf16(weight));
        float32x4_t v4lower = vcvt_f32_bf16(vld1_bf16(weight+4));

        float32x4_t v5upper = vaddq_f32(vmulq_f32(v3upper, v4upper), vcvt_f32_bf16(vld1_bf16(bias)));
        float32x4_t v5lower = vaddq_f32(vmulq_f32(v3lower, v4lower), vcvt_f32_bf16(vld1_bf16(bias+4)));

        vst1q_bf16(output, vcombine_bf16(vcvt_bf16_f32(v5lower), vcvt_bf16_f32(v5upper)));
        // store
    }
)

// lerp

// float weight = ((float *)w)[i % loopsize];
            // ((float *)output)[i] = ((float *)B)[i] * weight + ((float *)A)[i] * (1 - weight);

AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_lerp(float* input, float* other, float* weight, float* output){
        _mm512_storeu_ps(output, _mm512_fmadd_ps(_mm512_loadu_ps(input), _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_loadu_ps(weight)), _mm512_mul_ps(_mm512_loadu_ps(other), _mm512_loadu_ps(weight))));
    }

    __attribute__ ((target ("avx2")))
    void simd_lerp(float* input, float* other, float* weight, float* output){
        _mm256_storeu_ps(output, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(input), _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(weight))), _mm256_mul_ps(_mm256_loadu_ps(other), _mm256_loadu_ps(weight))));
    }

    __attribute__ ((target ("default")))
    void simd_lerp(float* input, float* other, float* weight, float* output){
        *output = *input * (1.0f - *weight) + *other * *weight;
    }
)

ARMONLY(
    void simd_lerp(float* input, float* other, float* weight, float* output){
        // arm neon
        float32x4_t v1 = vld1q_f32(input);
        float32x4_t v2 = vsubq_f32(vdupq_n_f32(1.0f), vld1q_f32(weight));
        float32x4_t v3 = vmulq_f32(v1, v2);
        float32x4_t v4 = vmulq_f32(vld1q_f32(other), vld1q_f32(weight));
        float32x4_t v5 = vaddq_f32(v3, v4);
        vst1q_f32(output, v5);
    }
)

// lerp bf16

AVXONLY(
    __attribute__ ((target ("avx512f")))
    void simd_lerp_bf16(bfloat16* input, bfloat16* other, bfloat16* weight, bfloat16* output){
        auto A = _mm512_loadu_si512(input);
        auto B = _mm512_loadu_si512(other);
        auto w = _mm512_loadu_si512(weight);

        __m512 v1u = bf16_float32(_mm512_extracti64x4_epi64(A, 0));
        __m512 v1l = bf16_float32(_mm512_extracti64x4_epi64(A, 1));

        __m512 v2u = bf16_float32(_mm512_extracti64x4_epi64(B, 0));
        __m512 v2l = bf16_float32(_mm512_extracti64x4_epi64(B, 1));

        __m512 v3u = bf16_float32(_mm512_extracti64x4_epi64(w, 0));
        __m512 v3l = bf16_float32(_mm512_extracti64x4_epi64(w, 1));

        __m512 v4u = _mm512_sub_ps(_mm512_set1_ps(1.0f), v3u);
        __m512 v4l = _mm512_sub_ps(_mm512_set1_ps(1.0f), v3l);

        __m512 v5u = _mm512_mul_ps(v1u, v4u);
        __m512 v5l = _mm512_mul_ps(v1l, v4l);

        __m512 v6u = _mm512_mul_ps(v2u, v3u);
        __m512 v6l = _mm512_mul_ps(v2l, v3l);

        __m512 v7u = _mm512_add_ps(v5u, v6u);
        __m512 v7l = _mm512_add_ps(v5l, v6l);

        _mm512_storeu_si512(output, float32_bf16(v7l, v7u));

    }

    __attribute__ ((target ("avx2")))
    void simd_lerp_bf16(bfloat16* input, bfloat16* other, bfloat16* weight, bfloat16* output){
        auto A = _mm256_loadu_si256((__m256i*)input);
        auto B = _mm256_loadu_si256((__m256i*)other);
        auto w = _mm256_loadu_si256((__m256i*)weight);

        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(A, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(A, 1));

        __m256 v2u = bf16_float32_avx2(_mm256_extracti128_si256(B, 0));
        __m256 v2l = bf16_float32_avx2(_mm256_extracti128_si256(B, 1));

        __m256 v3u = bf16_float32_avx2(_mm256_extracti128_si256(w, 0));
        __m256 v3l = bf16_float32_avx2(_mm256_extracti128_si256(w, 1));

        __m256 v4u = _mm256_sub_ps(_mm256_set1_ps(1.0f), v3u);
        __m256 v4l = _mm256_sub_ps(_mm256_set1_ps(1.0f), v3l);

        __m256 v5u = _mm256_mul_ps(v1u, v4u);
        __m256 v5l = _mm256_mul_ps(v1l, v4l);

        __m256 v6u = _mm256_mul_ps(v2u, v3u);
        __m256 v6l = _mm256_mul_ps(v2l, v3l);

        __m256 v7u = _mm256_add_ps(v5u, v6u);
        __m256 v7l = _mm256_add_ps(v5l, v6l);

        _mm256_storeu_si256((__m256i*)output, float32_bf16_avx2(v7l, v7u));
    }

    __attribute__ ((target ("default")))
    void simd_lerp_bf16(bfloat16* input, bfloat16* other, bfloat16* weight, bfloat16* output){
        *(output) = bfloat16(float(*(input)) * (1.0f - float(*(weight))) + float(*(other)) * float(*(weight)));
    }

)

ARMONLY(
    __attribute__ ((target ("bf16")))
    void simd_lerp_bf16(bfloat16* input, bfloat16* other, bfloat16* weight, bfloat16* output){
        // arm neon
        auto A = vld1q_bf16(input);
        auto B = vld1q_bf16(other);
        auto w = vld1q_bf16(weight);

        auto v1u = vcvt_f32_bf16(vget_high_bf16(A));
        auto v1l = vcvt_f32_bf16(vget_low_bf16(A));

        auto v2u = vcvt_f32_bf16(vget_high_bf16(B));
        auto v2l = vcvt_f32_bf16(vget_low_bf16(B));

        auto v3u = vcvt_f32_bf16(vget_high_bf16(w));
        auto v3l = vcvt_f32_bf16(vget_low_bf16(w));

        auto v4u = vsubq_f32(vdupq_n_f32(1.0f), v3u);
        auto v4l = vsubq_f32(vdupq_n_f32(1.0f), v3l);

        auto v5u = vmulq_f32(v1u, v4u);
        auto v5l = vmulq_f32(v1l, v4l);

        auto v6u = vmulq_f32(v2u, v3u);
        auto v6l = vmulq_f32(v2l, v3l);

        auto v7u = vaddq_f32(v5u, v6u);
        auto v7l = vaddq_f32(v5l, v6l);

        vst1q_bf16(output, vcombine_bf16(vcvt_bf16_f32(v7l), vcvt_bf16_f32(v7u)));        // store
    }
)

AVXONLY(
    // avx2 bf16 dot product
    

    __attribute__ ((target ("avx2","fma"),))
    __m256 dotbf16_avx2(__m256 acc, void* a, void* b){
        __m256i v1 = _mm256_loadu_si256((__m256i*)a);
        __m256i v2 = _mm256_loadu_si256((__m256i*)b);

        __m256 v1u = bf16_float32_avx2(_mm256_extracti128_si256(v1, 0));
        __m256 v1l = bf16_float32_avx2(_mm256_extracti128_si256(v1, 1));

        __m256 v2u = bf16_float32_avx2(_mm256_extracti128_si256(v2, 0));
        __m256 v2l = bf16_float32_avx2(_mm256_extracti128_si256(v2, 1));

        __m256 v3u = _mm256_fmadd_ps(v1u, v2u, acc);
        return _mm256_fmadd_ps(v1l, v2l, v3u);

    }

    __attribute__ ((target ("avx512bf16", "avx512vl")))
    __m256 dotbf16_avx2(__m256 acc, void* a, void* b){
        return _mm256_dpbf16_ps(acc, (__m256bh)_mm256_loadu_si256((__m256i*)a), (__m256bh)_mm256_loadu_si256((__m256i*)b));
    }
        

)


#endif // INTRINSICS_H