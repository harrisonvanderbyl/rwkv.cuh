#ifndef NORMALIZE_CPU_H
#define NORMALIZE_CPU_H

#include "tensor/tensor.h"
#include "tensor/intrinsics/intrinsics.h"

void normalize_cpu_kernel(void *input, void *weight, void *bias, void *output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype)
{

    size_t simdwidth = get_simd_width();
    for (size_t i = 0; i < size; i += headshape)
    {

        float mean = 0;
        float var = 0;

        if (dtype == TENSORTYPE::kFLOAT_32)
        {

            for (size_t j = i; j < i + headshape; j += simdwidth)
            {
                mean += simd_accumulate(flp(input) + j);
            }
            mean /= headshape;

            for (size_t j = i; j < i + headshape; j += simdwidth)
            {
                var += simd_variance_acc(flp(input) + j, mean);
            }

            var /= headshape;

            float vareps = sqrt(var + eps);

            for (size_t j = i; j < i + headshape; j++)
            {
                simd_norm_assign(flp(input) + j, mean, vareps, flp(weight) + j % lastshape, flp(bias) + j % lastshape, flp(output) + j);
            }
        }
        else if (dtype == TENSORTYPE::kBFLOAT_16)
        {
            for (size_t j = i; j < i + headshape; j += simdwidth * 2)
            {
                mean += simd_accumulate_bf16(bflp(input) + j);
            }
            mean /= headshape;

            for (size_t j = i; j < i + headshape; j += simdwidth * 2)
            {
                var += simd_variance_acc_bf16(bflp(input) + j, mean);
            }

            var /= headshape;

            float vareps = sqrt(var + eps);

            for (size_t j = i; j < i + headshape; j += simdwidth * 2)
            {
                simd_norm_assign_bf16(bflp(input) + j, mean, vareps, bflp(weight) + j % lastshape, bflp(bias) + j % lastshape, bflp(output) + j);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported datatype for normalize, only float32 and bfloat16 are supported on CPU");
        }
    }
}

#endif // NORMALIZE_CPU_H