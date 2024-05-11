#ifndef UINT8THREADING_H
#define UINT8THREADING_H
#include <atomic>
#include <thread>
#include <iostream>
// include threads

#include "tensor/tensor.h"
#include "tensor/operators/matmul/cpu.h"

void matmul8_cpu_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE)
{

    ThreadPool *pool = get_threadpool();

    auto headsize = OUTSHAPE / pool->heads;
    for (size_t head = 0; head < pool->heads; head++)
    {
        pool->add_job(
            [A, B, C, Ao, Ar, BBT, INSHAPE, OUTSHAPE, headsize, head]()
            {
                for (size_t bbt = 0; bbt < BBT; bbt += 1)
                {
                    float ss2f = sum_floats(flp(B) + bbt * INSHAPE, INSHAPE);

                    const auto BAINSHAPE = flp(B) + bbt * INSHAPE;

                    for (size_t b = head*headsize; b < (head + 1)*headsize; b += 1)
                    {
                        auto zz1 = dot_uint8_floats(A + b*INSHAPE, BAINSHAPE, INSHAPE);

                        (flp(C) + bbt * OUTSHAPE)[b] += ss2f * flp(Ao)[b] + zz1 * flp(Ar)[b];
                    }
                } },
            head);
    }
}

void matmul_cpu_kernal(void *A, void *B, void *C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype)
{
}

void wkv5_cpu_kernel(void *kk, void *vv, void *ww, void *uu, void *rr, void *ss, void *out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype)
{

    auto CH = C / H;

    uint32_t bsize = H * T * CH;

    // 1d tensor
    uint32_t tsize = H * CH;
    // 2d tensor
    uint32_t ttsize = H * CH * CH;

    // 1d
    uint32_t hsize = CH;
    // 2d
    uint32_t hhsize = CH * CH;

    ThreadPool *pool = get_threadpool();

    // size_t simdwidth = get_simd_width();
    for (uint32_t hh = 0; hh < H; hh++)
    {
        pool->add_job([kk, vv, ww, uu, rr, ss, out, T, B, C, H, CH, bsize, tsize, ttsize, hsize, hhsize, hh, dtype]
                      {
            for (uint32_t bb = 0; bb < B; bb++)
            {
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

                        float kkk = flp(kk)[iind];
                        float uuu = flp(uu)[hoffseti];
                        float rrr = flp(rr)[iind];
                        float www = flp(ww)[hoffseti];
                        
                       simd_wkv(CH, bhofseti, bbhhofseti, vv, ss, kkk, uuu, www, rrr, out);
                        
                    }
                }
            } }, hh);
    }
}

#endif // UINT8THREADING_H