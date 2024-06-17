
#ifndef tanh_NAIVE_H
#define tanh_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"

#include "tensor/operators/threading/threading.h"
void tanh_cpu_kernel(void *input, size_t size, TENSORTYPE dtype, size_t dims)
{

    size_t simdwidth = get_simd_width();

    auto pool = get_threadpool();

    auto mheads = pool->heads;

    while(dims%(mheads*8) !=0){
        mheads -=1;
    }

    if(mheads == 0){
        std::cout << "weird model, seek help\n";
        exit(1);
    }

    auto headsize = dims / mheads;

    pool->sync();


    if (dtype == TENSORTYPE::kFLOAT_32)
    {

        for (size_t t = 0; t < mheads; t++)
        {
            pool->add_job([input, size, dims, simdwidth, t, headsize]
                          {
                for (size_t ii = t * headsize; ii < size; ii += dims)
                {
                    for (size_t i = ii; i < ii + headsize; i += simdwidth)
                    {
                        simd_tanh(flp(input) + i);}
                } }, t);
        }
    }
    else
    {
        throw std::runtime_error("tanh only implemented for float");
    }
    pool->sync();
}

#endif // tanh_NAIVE_H