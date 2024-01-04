#ifndef NORMALIZEDISPACH_H
#define NORMALIZEDISPACH_H

#include "tensor/tensor.h"

void normalize_cuda_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype);
void normalize_cpu_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype);
// layernorm kernel


inline Tensor Tensor::normalize(Tensor& weight, Tensor& bias, Tensor& result, size_t heads, float eps)
{

    auto lastshape = this->shape.back();
    auto headshape = lastshape / heads;

    if (this->device == DEVICE::CPU){

        normalize_cpu_kernel(this->data, weight.data, bias.data, result.data, eps, lastshape, headshape, this->get_element_count(), this->dtype);

    } else CUDAONLY
    {        
        normalize_cuda_kernel(this->data, weight.data, bias.data, result.data, eps, lastshape, headshape, this->get_element_count(), this->dtype);
    }

    return result;

}









#endif // NORMALIZEDISPACH_H