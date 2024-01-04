#ifndef TENSOR_CPU_MATMUL_H
#define TENSOR_CPU_MATMUL_H

#include <iostream>
#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"
#include "tensor/operators/matmul/threading.h"


    __attribute__((target("avx2","fma"))) 
    void dopartial(MatMulJob job)
    {
        // do the work
        auto A = job.A;
        auto B = job.B;
        auto C = job.C;
        auto INSHAPE = job.INSHAPE;
        auto OUTSHAPE = job.OUTSHAPE;
        auto Ao = job.Ao;
        auto Ar = job.Ar;
        auto Batch = job.bbt;
        auto ii = job.ii;

        for (size_t bbt = 0; bbt < Batch; bbt += 1)
        {
            for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
            {
                for (size_t b = 0; b < 16; b += 8)
                {
                    const float *Ario1 = (((float *)Ar) + dii + b);
                    const float *Aoio1 = (((float *)Ao) + dii + b);

                    auto zz1 = _mm256_loadu_ps(flp(C) + bbt * OUTSHAPE + dii + b);

                    for (uint32_t i = dii + b; i < dii + b + 8; i += 1)
                    {
                        auto Aoio = Aoio1[i & 7]/Ario1[i & 7];

                        const auto IAINSHAPE = A + i * INSHAPE;

                        auto sum1 = _mm256_set1_ps(0.0);
                        auto s2 = _mm256_set1_ps(0.0);
                        for (uint32_t k = 0; k < INSHAPE; k += 8)
                        {
                            // avx2
                            auto w = _mm256_cvtepu8_epi32(_mm_lddqu_si128((__m128i *)(IAINSHAPE + k))); // Load the input uint8_t vector
                            auto u = _mm256_cvtepi32_ps(w) ; // Convert uint32_t to float32_t
                            auto v = _mm256_load_ps(((float *)B) + bbt * INSHAPE + k);
                            sum1 = _mm256_fmadd_ps(u, v, sum1);
                            s2 = _mm256_add_ps(s2,v);
                        }

                        sum1 += s2 * + Aoio;

                        zz1[i & 7] += (sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7])*Ario1[i & 7];
                    }

                    _mm256_store_ps(
                        flp(C) + bbt * OUTSHAPE + dii + b,
                        zz1);
                }
            }
        }
    }
 



__attribute__((target("avx2","fma")))
void dopartialfp(MatMulJob job)
{
    // do the work

    auto A = job.Ao;
    auto B = job.B;
    auto C = job.C;
    auto INSHAPE = job.INSHAPE;
    auto OUTSHAPE = job.OUTSHAPE;
    auto Batch = job.bbt;
    auto ii = job.ii;
    auto dtype = job.dtype;

    if (dtype == TENSORTYPE::kFLOAT_32){
        for (size_t bbt = 0; bbt < Batch; bbt += 1)
        {
            for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
            {
                for (size_t b = 0; b < 16; b += 8)
                {
                    auto zz1 = _mm256_loadu_ps(flp(C) + bbt * OUTSHAPE + dii + b);

                    for (uint32_t i = dii + b; i < dii + b + 8; i += 1)
                    {
                        const float* IAINSHAPE = flp(A) + i * INSHAPE;

                        auto sum1 = _mm256_set1_ps(0.0);
                        auto sum2 = _mm256_set1_ps(0.0);
                        for (uint32_t k = 0; k < INSHAPE; k += 16)
                        {
                            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(IAINSHAPE + k), _mm256_loadu_ps(flp(B) + bbt * INSHAPE + k), sum1);
                            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(IAINSHAPE + k + 8), _mm256_loadu_ps(flp(B) + bbt * INSHAPE + k + 8), sum2);
                        }

                        sum1 = _mm256_add_ps(sum1, sum2);

                        zz1[i & 7] += sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
                    }

                    _mm256_storeu_ps(
                        flp(C) + bbt * OUTSHAPE + dii + b,
                        zz1);
                }
            }
        }
    }else{
         for (size_t bbt = 0; bbt < Batch; bbt += 1)
        {
            for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
            {
                
                    float zz1[16] = {0.0};
                    _mm256_storeu_ps(
                        &zz1[0],
                        bf16_float32_avx2(
                            _mm_loadu_si128((__m128i_u*)(bflp(C) + bbt * OUTSHAPE + dii))
                        )
                        );
                    _mm256_storeu_ps(
                        &zz1[8],
                        bf16_float32_avx2(
                            _mm_loadu_si128((__m128i_u*)(bflp(C) + bbt * OUTSHAPE + dii + 8))
                        )
                    );


                    for (uint32_t i = dii; i < dii + 16; i += 1)
                    {
                        const bfloat16* IAINSHAPE = bflp(A) + i * INSHAPE;

                        auto sum1 = _mm256_set1_ps(0.0);
                        for (uint32_t k = 0; k < INSHAPE; k += 16)
                        {
                            sum1 = dotbf16_avx2(sum1, (void*)(IAINSHAPE + k), bflp(B) + bbt * INSHAPE + k);
                        }

                        zz1[i & 15] += sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
                    }

                    __m256 values = _mm256_loadu_ps(&zz1[0]);
                    __m256 values2 = _mm256_loadu_ps(&zz1[8]);

                    auto values3 = float32_bf16_avx2(values2, values);

                    _mm256_storeu_si256((__m256i*)(bflp(C) + bbt * OUTSHAPE + dii), values3);

            
            }
        }
    }

}


__attribute__((target("default")))
void dopartialfp(MatMulJob job)
{
printf("dopartialfp not implemented for this architecture");
}




ARMONLY(
    
    __attribute__((target("neon")))
    void dopartial(MatMulJob job)
    {
        // do the work
        auto A = job.A;
        auto B = job.B;
        auto C = job.C;
        auto INSHAPE = job.INSHAPE;
        auto OUTSHAPE = job.OUTSHAPE;
        auto Ao = job.Ao;
        auto Ar = job.Ar;
        auto Batch = job.bbt;
        auto ii = job.ii;
        for (size_t bbt = 0; bbt < Batch; bbt += 1)
        {
            for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
            {
                for (size_t b = 0; b < 16; b += 4)
                {
                    const auto Ario1 = *(Ar + dii + b);
                    const auto Aoio1 = *(Ao + dii + b);

                    // set zero neon
                    auto zz1 = vdupq_n_f32(0.0);

                    for (uint32_t i = dii + b; i < dii + b + 4; i += 1)
                    {
                        auto Aoio = Aoio1[i & 3];

                        const auto IAINSHAPE = A + i * INSHAPE;

                        auto sum1 = vdupq_n_f32(0.0);
                        auto sum2 = vdupq_n_f32(0.0);
                        // remember to change this to load from residual

                        for (uint32_t k = 0; k < INSHAPE; k += 8)
                        {

                            auto u16_vec = vmovl_u8(vld1_u8((IAINSHAPE + k)));

                            // Convert uint8_t values to float32x4_t
                            // convert uint8_t to uint16_t
                            auto u32_low_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec))) + Aoio;   // Extract lower part and convert to uint32_t
                            auto u32_high_vec = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec))) + Aoio; // Extract upper part and convert to uint32_t
                            // Load the input float vector
                            // Perform the multiplication with inp vector
                            sum1 = vmlaq_f32(u32_low_vec, vld1q_f32(B + bbt * INSHAPE + k), sum1);
                            sum2 = vmlaq_f32(u32_high_vec, vld1q_f32(B + bbt * INSHAPE + k + 4), sum2);
                        }

                        sum1 = vaddq_f32(sum1, sum2);

                        zz1[i & 3] = zz1[i & 3] + sum1[0] + sum1[1] + sum1[2] + sum1[3];
                    }

                    vst1q_f32(
                        (C + bbt * OUTSHAPE + dii + b),
                        zz1 * Ario1);
                }
            }
        }
    }

)


void dopartialwkv5att(MatMulJob job)
{
    auto T = job.bbt;
    auto CH = job.INSHAPE;
    auto bb = job.OUTSHAPE;
    auto kk = job.Ao;
    auto ww = job.C;
    auto vv = job.Ar;
    auto uu = job.Bt;
    auto rr = job.Ct;
    auto ss = job.ex;
    auto out = job.B;
    auto H = job.H;
    auto hh = job.hh;
    auto dtype = job.dtype;

    // 1d
    uint32_t bsize = H * T * CH;

    // 1d tensor
    uint32_t tsize = H * CH;
    // 2d tensor
    uint32_t ttsize = H * CH * CH;

    // 1d
    uint32_t hsize = CH;
    // 2d
    uint32_t hhsize = CH * CH;

    // size_t simdwidth = get_simd_width();

    for (uint32_t t = 0; t < T; t++)
    {
        for (uint32_t i = 0; i < CH; i++)
        {
            auto btimeoffset = bb * bsize;
            auto timeoffset = btimeoffset + t * tsize;
            auto bbhsize = bb * ttsize;

            auto hoffset = hh * hsize;
            auto bhofseti = timeoffset + hoffset;
            auto bbhhsize = bbhsize + hh * hhsize;

            uint32_t iind = bhofseti + i;
            auto hoffseti = hoffset + i;
            auto bbhhofseti = bbhhsize + i * hsize;

            float kkk;
            float uuu;
            float rrr;
            float www;
            // auto kkk = kk[iind];
            if (dtype == TENSORTYPE::kFLOAT_32)
            {
                kkk = flp(kk)[iind];
                uuu = flp(uu)[hoffseti];
                rrr = flp(rr)[iind];
                www = flp(ww)[hoffseti];
            }
            if (dtype == TENSORTYPE::kBFLOAT_16)
            {
                kkk = float(bflp(kk)[iind]);
                uuu = float(bflp(uu)[hoffseti]);
                rrr = float(bflp(rr)[iind]);
                www = float(bflp(ww)[hoffseti]);
            }

            for (uint32_t j = 0; j < CH; j += 1)
            {
                uint32_t jind = bhofseti + j;
                uint32_t sind = bbhhofseti + j;

                // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                float vvv;
                if (dtype == TENSORTYPE::kFLOAT_32)
                {
                    vvv = flp(vv)[jind];
                }
                if (dtype == TENSORTYPE::kBFLOAT_16)
                {
                    vvv = float(bflp(vv)[jind]);
                }

                float sss = flp(ss)[sind];

                // multiply kkk and vvv
                auto atu = vvv * kkk;

                // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                auto sssatuuuu = atu * uuu + sss;

                float outf = sssatuuuu * rrr;

                // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                if (dtype == TENSORTYPE::kBFLOAT_16)
                {
                    *(bflp(out) + jind) = float32_to_bfloat16(outf + bfloat16_to_float32(*(bflp(out) + jind)));
                }
                if (dtype == TENSORTYPE::kFLOAT_32)
                {
                    flp(out)[jind] += outf;
                }

                *(flp(ss) + sind) = sss * www + atu;
            }
        }
    }
}

#endif // TENSOR_CPU_MATMUL_H