#ifndef NORMALIZE_CPU_H
#define NORMALIZE_CPU_H

#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"

#include "tensor/operators/threading/threading.h"
static void normalize_cpu_kernel(void *input, void *weight, void *bias, void *output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype)
{
    auto threadpool = get_threadpool(1);
    size_t simdwidth = get_simd_width();

    size_t newheadshape = lastshape / threadpool->heads;
    if (headshape % newheadshape != 0 and newheadshape % headshape != 0)
    {
        throw std::runtime_error("Headshape and threads must share common factor");
    }

    threadpool->sync();

    for (size_t oo = 0; oo < threadpool->heads; oo += 1)
    {
        threadpool->add_job(
            [input, weight, bias, output, eps, lastshape, headshape, dtype, simdwidth, size, oo, newheadshape]
            {
                for (size_t ii = oo * newheadshape; ii < size; ii += lastshape)
                {
                    for (size_t i = ii; i < ii + newheadshape; i += headshape)
                    {
                        float mean = 0;
                        float var = 0;

                        size_t startOfActualHead = i - (i % headshape);

                        if (dtype == TENSORTYPE::kFLOAT_32)
                        {

                            for (size_t j = startOfActualHead; j < startOfActualHead + headshape; j += simdwidth)
                            {
                                mean += simd_accumulate(flp(input) + j);
                            }
                            mean /= headshape;

                            for (size_t j = startOfActualHead; j < startOfActualHead + headshape; j += simdwidth)
                            {
                                var += simd_variance_acc(flp(input) + j, mean);
                            }

                            var /= headshape;

                            float vareps = sqrt(var + eps);

                            for (size_t j = i; j < i + std::min(newheadshape,headshape); j += simdwidth)
                            {
                                simd_norm_assign(flp(input) + j, mean, vareps, flp(weight) + (j % lastshape), flp(bias) + (j % lastshape), flp(output) + j);
                            }
                        }
                        else
                        {
                            throw std::runtime_error("Unsupported datatype for normalize, only float32 supported on CPU, dtype is " + get_dtype_name(dtype));
                        }
                    }
                }
            },
            oo);
    }

    

}

#endif // NORMALIZE_CPU_H