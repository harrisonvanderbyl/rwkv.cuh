#ifndef TENSOR_CPU_MATMUL_H
#define TENSOR_CPU_MATMUL_H
#include <atomic>
#include <thread>
#include <iostream>
// include threads

#include "tensor/tensor.h"
#include "tensor/operators/matmul/cpu.h"

#include "tensor/operators/threading/threading.h"

void matmul8_cpu_kernal(uint8_t *A, void *B, void *C, void *Ao, void *Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, MMACTFUNC func)
{

    ThreadPool *pool = get_threadpool();

    auto headsize = OUTSHAPE / pool->heads;
    for (size_t head = 0; head < pool->heads; head++)
    {
        pool->add_job(
            [A, B, C, Ao, Ar, BBT, INSHAPE, OUTSHAPE, headsize, head, func]()
            {
                for (size_t bbt = 0; bbt < BBT; bbt += 1)
                {
                    float ss2f = sum_floats(flp(B) + bbt * INSHAPE, INSHAPE);

                    const auto BAINSHAPE = flp(B) + bbt * INSHAPE;
                    auto spot =flp(C) + bbt * OUTSHAPE;
                    for (size_t b = head*headsize; b < (head + 1)*headsize; b += 1)
                    {


                        double zz1 = dot_uint8_floats(A + b*INSHAPE, BAINSHAPE, INSHAPE);
                        
                        zz1 = ss2f * (((float*)(Ao))[b]) + zz1 * (((float*)(Ar))[b]);
                        if(func == TANH){
                            auto ax = exp(spot[b]+ zz1);
                            auto bx = exp(-(spot[b]+zz1));
                            spot[b] = (ax-bx)/(ax+bx);
                        }
                        if(func == RELUSQUARE){
                            spot[b] += zz1;
                            spot[b] = (spot[b]*fmaxf(spot[b],0.0f));
                        }
                        if(func == SWISHMUL){
                            spot[b] = (spot[b] * zz1)/(1.0 + exp(-zz1));
                        }
                        if(func == SIGMOIDMUL){
                            spot[b] = ((flp(B)-BBT*INSHAPE + bbt*OUTSHAPE)[b])/(1.0 + exp(-zz1)) + spot[b];
                        }
                        if(func == NONE){
                            spot[b] += zz1;
                        }
                        if(func == EXPNEGEXP){
                            spot[b] = exp(-exp(zz1+spot[b]));
                        }
                        if(func == SETVALUE){
                            spot[b] = zz1;
                        }
                    }
                } },
            head);
    }
}

void matmul_cpu_kernal(void *A, void *B, void *C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype, size_t bmmshape, MMACTFUNC func)
{
    ThreadPool *pool = get_threadpool();

    // std::cout << INSHAPE<<"<in->out>" << OUTSHAPE << "\n";

    auto headsize = (OUTSHAPE) / pool->heads;
    auto headsnum = pool->heads;
    // assert((headsize % get_simd_width()) == 0);

    // assert((headsize%get_simd_width()) == 0); // use different thread counts, try a different amount of threads, ie 4 for 1b5 should work
    for (size_t head = 0; head < headsnum; head++)
    {
        pool->add_job(
            [A, B, C, BBT, INSHAPE, OUTSHAPE, headsize, head, bmmshape, func]()
            {
                for (size_t bbt = 0; bbt < BBT; bbt += 1)
                {
                    for (size_t bmmindex = 0; bmmindex < bmmshape; bmmindex += 1){

                    const auto BAINSHAPE = flp(B) + bbt * (INSHAPE*bmmshape) + bmmindex * INSHAPE;
                    auto spot = flp(C) + bmmindex * BBT * OUTSHAPE + bbt * OUTSHAPE;
                    for (size_t b = head*headsize; b < (head + 1)*headsize; b += 1)
                    {
                        double zz1 = dot_floats(flp(A) + b*INSHAPE + bmmindex * INSHAPE*OUTSHAPE, BAINSHAPE, INSHAPE);
                        
                        // spot[b] =  zz1;
                        if(func == TANH){
                            auto ax = exp(spot[b]+zz1);
                            auto bx = exp(-(spot[b]+zz1));
                            spot[b] = (ax-bx)/(ax+bx);
                        }
                        if(func == RELUSQUARE){
                            spot[b] += zz1;
                            spot[b] = (spot[b]*fmaxf(spot[b],0.0f));
                        }
                        if(func == SWISHMUL){
                            spot[b] = (spot[b] * zz1)/(1.0 + exp(-zz1));
                        }
                        if(func == SIGMOIDMUL){
                            spot[b] = ((flp(B)-BBT*INSHAPE + bbt*OUTSHAPE)[b])/(1.0 + exp(-zz1)) + spot[b];
                        }
                        if(func == SETVALUE){
                            spot[b] = zz1;
                        }
                        if(func == NONE){
                            spot[b] += zz1;
                        }
                        if(func == EXPNEGEXP){
                            spot[b] = exp(-exp(zz1+spot[b]));
                        }
                        
                    }
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


#endif // TENSOR_CPU_MATMUL_H