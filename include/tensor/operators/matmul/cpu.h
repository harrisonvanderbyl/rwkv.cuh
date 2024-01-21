#ifndef TENSOR_CPU_MATMUL_H
#define TENSOR_CPU_MATMUL_H

#include <iostream>
#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"
#include "tensor/operators/matmul/threading.h"
AVXONLY(
    void dopartial(MatMulJob job) {
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
            auto s2 = _mm256_set1_ps(0.0);
            
            for (uint32_t k = 0; k < INSHAPE; k += 8)
                        {
                            s2 = _mm256_add_ps(s2, _mm256_load_ps(flp(B) + bbt * INSHAPE + k));
                        }
            auto ss2m = _mm_add_ps(_mm256_extractf128_ps(s2,0),_mm256_extractf128_ps(s2,1));
            float* ss2 = (float*)&ss2m;
            float ss2f = ss2[0] + ss2[1] + ss2[2] + ss2[3];

            const auto BAINSHAPE = flp(B) + bbt * INSHAPE;

            for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
            {
                for (size_t b = 0; b < 16; b += 8)
                {

                    auto zz1 = __m256{0.0};
                    float* rzz = (float*)&zz1;

                    for (auto IAINSHAPE = A + (dii + b) * INSHAPE; IAINSHAPE < A + (dii + b + 8) * INSHAPE; IAINSHAPE += INSHAPE)
                    {
                        auto sum1 = _mm256_set1_ps(0.0);
                        for (auto k = IAINSHAPE; k < IAINSHAPE + INSHAPE; k += 8)
                        { 
                            sum1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_lddqu_si128((__m128i *)(k)))), _mm256_load_ps(BAINSHAPE + (k-IAINSHAPE)), sum1);
                        }
                        float* sum1l = (float*)&sum1;
                        *(rzz++) += sum1l[0] + sum1l[1] + sum1l[2] + sum1l[3] + sum1l[4] + sum1l[5] + sum1l[6] + sum1l[7];
                        
                    }

                    _mm256_storeu_ps(
                        flp(C) + bbt * OUTSHAPE + dii + b,
                        _mm256_fmadd_ps(_mm256_set1_ps(ss2f), _mm256_load_ps(flp(Ao) + dii + b) , _mm256_fmadd_ps(zz1 , _mm256_load_ps(flp(Ar) + dii + b), _mm256_load_ps(flp(C) + bbt * OUTSHAPE + dii + b))));
                }
            }
        }
    }


    void dopartialfp(MatMulJob job) {
        // do the work

        auto A = job.Ao;
        auto B = job.B;
        auto C = job.C;
        auto INSHAPE = job.INSHAPE;
        auto OUTSHAPE = job.OUTSHAPE;
        auto Batch = job.bbt;
        auto ii = job.ii;
        auto dtype = job.dtype;

        if (dtype == TENSORTYPE::kFLOAT_32)
        {
            for (size_t bbt = 0; bbt < Batch; bbt += 1)
            {
                for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
                {
                    for (size_t b = 0; b < 16; b += 8)
                    {
                        auto zz1 = _mm256_loadu_ps(flp(C) + bbt * OUTSHAPE + dii + b);

                        for (uint32_t i = dii + b; i < dii + b + 8; i += 1)
                        {
                            const float *IAINSHAPE = flp(A) + i * INSHAPE;

                            auto sum1 = _mm256_set1_ps(0.0f);
                            auto sum2 = _mm256_set1_ps(0.0f);
                            for (uint32_t k = 0; k < INSHAPE; k += 16)
                            {
                                sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(IAINSHAPE + k), _mm256_loadu_ps(flp(B) + bbt * INSHAPE + k), sum1);
                                sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(IAINSHAPE + k + 8), _mm256_loadu_ps(flp(B) + bbt * INSHAPE + k + 8), sum2);
                            }

                            sum1 = _mm256_add_ps(sum1, sum2);
                            float* sum1l = (float*)&sum1;
                            flp(&zz1)[i & 7] += sum1l[0] + sum1l[1] + sum1l[2] + sum1l[3] + sum1l[4] + sum1l[5] + sum1l[6] + sum1l[7];
                        }

                        _mm256_storeu_ps(
                            flp(C) + bbt * OUTSHAPE + dii + b,
                            zz1);
                    }
                }
            }
        }
        else
        {
            for (size_t bbt = 0; bbt < Batch; bbt += 1)
            {
                for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
                {

                    float zz1[16] = {0.0};
                    _mm256_storeu_ps(
                        &zz1[0],
                        bf16_float32_avx2(
                            _mm_loadu_si128((__m128i *)(bflp(C) + bbt * OUTSHAPE + dii))));
                    _mm256_storeu_ps(
                        &zz1[8],
                        bf16_float32_avx2(
                            _mm_loadu_si128((__m128i *)(bflp(C) + bbt * OUTSHAPE + dii + 8))));

                    for (uint32_t i = dii; i < dii + 16; i += 1)
                    {
                        const bfloat16 *IAINSHAPE = bflp(A) + i * INSHAPE;

                        auto sum1 = _mm256_set1_ps(0.0);
                        for (uint32_t k = 0; k < INSHAPE; k += 16)
                        {
                            sum1 = dotbf16_avx2(sum1, (void *)(IAINSHAPE + k), bflp(B) + bbt * INSHAPE + k);
                        }
                        float* sum1l = (float*)&sum1;
                        zz1[i & 15] += sum1l[0] + sum1l[1] + sum1l[2] + sum1l[3] + sum1l[4] + sum1l[5] + sum1l[6] + sum1l[7];
                    }

                    __m256 values = _mm256_loadu_ps(&zz1[0]);
                    __m256 values2 = _mm256_loadu_ps(&zz1[8]);

                    auto values3 = float32_bf16_avx2(values2, values);

                    _mm256_storeu_si256((__m256i *)(bflp(C) + bbt * OUTSHAPE + dii), values3);
                }
            }
        }
    }

)
// ARMONLY(

//     __attribute__((target("bf16"))) void dopartialfp(MatMulJob job) {
//         // do the work

//         auto A = job.Ao;
//         auto B = job.B;
//         auto C = job.C;
//         auto INSHAPE = job.INSHAPE;
//         auto OUTSHAPE = job.OUTSHAPE;
//         auto Batch = job.bbt;
//         auto ii = job.ii;
//         auto dtype = job.dtype;

//         if (dtype == TENSORTYPE::kFLOAT_32)
//         {
//             for (size_t bbt = 0; bbt < Batch; bbt += 1)
//             {
//                 for (size_t dii = ii; dii < OUTSHAPE; dii += 16 * 16)
//                 {
//                     for (size_t b = 0; b < 16; b += 4)
//                     {
//                         auto zz1 = vld1q_f32(flp(C) + bbt * OUTSHAPE + dii + b);

//                         for (uint32_t i = dii + b; i < dii + b + 4; i += 1)
//                         {
//                             const float *IAINSHAPE = flp(A) + i * INSHAPE;

//                             auto sum1 = vdupq_n_f32(0.0);
//                             auto sum2 = vdupq_n_f32(0.0);
//                             for (uint32_t k = 0; k < INSHAPE; k += 8)
//                             {
//                                 sum1 = vmlaq_f32(vld1q_f32(IAINSHAPE + k), vld1q_f32(flp(B) + bbt * INSHAPE + k), sum1);
//                                 sum2 = vmlaq_f32(vld1q_f32(IAINSHAPE + k + 4), vld1q_f32(flp(B) + bbt * INSHAPE + k + 4), sum2);
//                             }

//                             sum1 = vaddq_f32(sum1, sum2);

//                             zz1[i & 3] += sum1[0] + sum1[1] + sum1[2] + sum1[3];
//                         }

//                         vst1q_f32(
//                             flp(C) + bbt * OUTSHAPE + dii + b,
//                             zz1);
//                     }
//                 }
//             }
//         }
//         else
//         {
//             //  for (size_t bbt = 0; bbt < Batch; bbt += 1)
//             // {
//             //     for (size_t dii = ii; dii < OUTSHAPE; dii += 4 * 16)
//             //     {

//             //             float zz1[4] = {0.0};
//             //             _mm256_storeu_ps(
//             //                 &zz1[0],
//             //                 bf16_float32_avx2(
//             //                     _mm_loadu_si128((__m128i_u*)(bflp(C) + bbt * OUTSHAPE + dii))
//             //                 )
//             //                 );

//             //             for (uint32_t i = dii; i < dii + 16; i += 1)
//             //             {
//             //                 const bfloat16* IAINSHAPE = bflp(A) + i * INSHAPE;

//             //                 auto sum1 = _mm256_set1_ps(0.0);
//             //                 for (uint32_t k = 0; k < INSHAPE; k += 16)
//             //                 {
//             //                     sum1 = dotbf16_avx2(sum1, (void*)(IAINSHAPE + k), bflp(B) + bbt * INSHAPE + k);
//             //                 }

//             //                 zz1[i & 15] += sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
//             //             }

//             //             __m256 values = _mm256_loadu_ps(&zz1[0]);
//             //             __m256 values2 = _mm256_loadu_ps(&zz1[8]);

//             //             auto values3 = float32_bf16_avx2(values2, values);

//             //             _mm256_storeu_si256((__m256i*)(bflp(C) + bbt * OUTSHAPE + dii), values3);

//             //     }
//             // }
//             std::cout << "Fix later, no bf16 matmul;";
//             exit(0);
//         }
//     }

//     void dopartial(MatMulJob job) {
//         // do the work
//         auto A = job.A;
//         auto B = job.B;
//         auto C = job.C;
//         auto INSHAPE = job.INSHAPE;
//         auto OUTSHAPE = job.OUTSHAPE;
//         auto Ao = job.Ao;
//         auto Ar = job.Ar;
//         auto Batch = job.bbt;
//         auto ii = job.ii;
//         for (size_t bbt = 0; bbt < Batch; bbt += 1)
//         {
//             for (size_t bii = ii; bii < OUTSHAPE; bii += 16 * 16)
//             {
//                 for (size_t i = bii; i < bii + 16; i += 1)
//                 {
//                     const auto Ario1 = *(flp(Ar) + i );
//                     const auto Aoio1 = *(flp(Ao) + i ) / Ario1;

//                     auto sum1 = vdupq_n_f32(0.0);
//                     auto sum2 = vdupq_n_f32(0.0);
//                     // remember to change this to load from residual

//                     for (uint32_t k = 0; k < INSHAPE; k += 8)
//                     {
                        
//                         auto u16_vec = vmovl_u8(vld1_u8((A + i * INSHAPE + k)));
//                         auto inp = vld1q_f32(flp(B) + bbt * INSHAPE + k);
//                         auto inp1 = vld1q_f32(flp(B) + bbt * INSHAPE + k + 4);

//                         // Convert uint8_t values to float32x4_t
//                         // convert uint8_t to uint16_t
//                         auto u32_low_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec))) + Aoio1;   // Extract lower part and convert to uint32_t
//                         auto u32_high_vec = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec))) + Aoio1; // Extract upper part and convert to uint32_t
//                         // Load the input float vector

//                         // Perform the multiplication with inp vector
//                         sum1 = vmlaq_f32(sum1, u32_low_vec, inp);
//                         sum2 = vmlaq_f32(sum2, u32_high_vec, inp1);
//                     }

//                     sum1 = vaddq_f32(sum1, sum2);

//                     *(flp(C) + bbt * OUTSHAPE + i ) += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * Ario1;
//                 }
//             }
//         }
//     })

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
    bool v6 = job.type == JOBTYPE::RWKV_ATT_V6;

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

            auto htimeoffset = t * hsize;
            auto thofoffseti = htimeoffset + hoffseti;
            auto bbhhofseti = bbhhsize + i * hsize;

            float kkk;
            float uuu;
            float rrr;
            float www;
            auto wind = hoffseti;
            if (v6){
                wind = iind;
            }
            // auto kkk = kk[iind];
            if (dtype == TENSORTYPE::kFLOAT_32)
            {
                kkk = flp(kk)[iind];
                uuu = flp(uu)[hoffseti];
                rrr = flp(rr)[iind];
                www = flp(ww)[wind] ;
            }
            if (dtype == TENSORTYPE::kBFLOAT_16)
            {
                uint16_t xx[4] = {((uint16_t *)(kk))[iind], ((uint16_t *)(uu))[hoffseti], (((uint16_t *)(rr))[iind]), (((uint16_t *)(ww))[iind])};
                uint32_t yy[4] = {uint32_t(xx[0]) << 16, uint32_t(xx[1]) << 16, uint32_t(xx[2]) << 16, uint32_t(xx[3]) << 16};
                kkk = flp(yy)[0];
                uuu = flp(yy)[1];
                rrr = flp(yy)[2];
                www = flp(yy)[3];
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
                    uint32_t xyy = uint32_t(*((uint16_t *)vv + jind)) << 16;
                    vvv = *(flp(&xyy));
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
                    auto outui = uint32_t(*(((uint16_t *)out) + jind)) << 16;
                    outf = (outf + *(float *)(&outui));
                    *(((uint16_t *)out) + jind) = uint16_t((*(uint32_t *)&outf) >> 16);
                }
                if (dtype == TENSORTYPE::kFLOAT_32)
                {
                    flp(out)[jind] += outf;
                }

                *(flp(ss) + sind) = sss * www + atu ;
            }
        }
    }
}

#endif // TENSOR_CPU_MATMUL_H