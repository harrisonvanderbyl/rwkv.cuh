#ifndef SIGMOIDMUL_NAIVE_H
#define SIGMOIDMUL_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"

#include "tensor/operators/threading/threading.h"
void sigmoidmul_cpu_kernel(void *input, void *other, void *residual, void *output, size_t size, TENSORTYPE dtype, size_t dims)
{

    size_t simdwidth = get_simd_width();

    auto pool = get_threadpool();

    auto headsize = dims / pool->heads;

    if (dtype == TENSORTYPE::kFLOAT_32)
    {

        for (size_t t = 0; t < pool->heads; t++)
        {
            pool->add_job([input, output, size, dims, simdwidth, t, headsize, other, residual]
                          {
                for (size_t ii = t * headsize; ii < size; ii += dims)
                {
                    for (size_t i = ii; i < ii + headsize; i += simdwidth)
                    {
            simd_sigmoidmul(flp(input) + i, flp(other) + i, flp(residual) + i, flp(output) + i);}
                } }, t);
        }
    }
    else
    {
        throw std::runtime_error("sigmoidmul only implemented for float");
    }
}

#endif // SIGMOIDMUL_NAIVE_H