#ifndef UINT8THREADING_H
#define UINT8THREADING_H
#include <atomic>
#include <thread>
#include <iostream>
// include threads

#include "tensor/tensor.h"
#include "tensor/operators/matmul/cpu.h"

#include "tensor/operators/threading/threading.h"

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
    ThreadPool *pool = get_threadpool();

    auto headsize = OUTSHAPE / pool->heads;
    for (size_t head = 0; head < pool->heads; head++)
    {
        pool->add_job(
            [A, B, C, BBT, INSHAPE, OUTSHAPE, headsize, head]()
            {
                for (size_t bbt = 0; bbt < BBT; bbt += 1)
                {

                    const auto BAINSHAPE = flp(B) + bbt * INSHAPE;

                    for (size_t b = head*headsize; b < (head + 1)*headsize; b += 1)
                    {
                        auto zz1 = dot_floats(flp(A) + b*INSHAPE, BAINSHAPE, INSHAPE);

                        (flp(C) + bbt * OUTSHAPE)[b] +=  zz1;
                    }
                } },
            head);
    }
}

void wkv5_cpu_kernel(void *kk, void *vv, void *ww, void *uu, void *rr, void *ss, void *out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype)
{

    ThreadPool *pool = get_threadpool();

    size_t headsize = H / pool->heads;

    for (uint32_t pid = 0; pid < pool->heads; pid++)
    {
        pool->add_job([vv, ss, kk, uu, ww, rr, out, T, B, C, H, headsize, pid]
                      {
            for (size_t head = pid * headsize; head < (pid + 1) * headsize; head++)
            {
                
                for (uint32_t bb = 0; bb < B; bb++)
                {
                    for (uint32_t t = 0; t < T; t++)
                    { 
                        
                                                
                        simd_wkv(B,T,H,C/H, bb, t, head, flp(vv), flp(ss), flp(kk), flp(uu), flp(ww), flp(rr), flp(out));
                            
                        
                    }
                }
            } }, pid);
    }
}

#endif // UINT8THREADING_H