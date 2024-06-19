
#ifndef tanh_NAIVE_H
#define tanh_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"

#include "tensor/operators/threading/threading.h"
void tanh_cpu_kernel(void *input, size_t size, TENSORTYPE dtype)
{

    size_t simdwidth = get_simd_width();

    auto pool = get_threadpool();

    auto mheads = pool->heads;

    auto headsize = size / mheads;

    pool->sync();


    if (dtype == TENSORTYPE::kFLOAT_32)
    {

        for (size_t t = 0; t < mheads; t++)
        {
            pool->add_job([input, size, simdwidth, t, headsize]
                          {
                
                    for (size_t i = t*headsize; i < (t+1)*headsize; i += simdwidth)
                    {
                        simd_tanh(flp(input) + i);}
                 }, t);
        }
    }
    else
    {
        throw std::runtime_error("tanh only implemented for float");
    }
    pool->sync();
}

#endif // tanh_NAIVE_H