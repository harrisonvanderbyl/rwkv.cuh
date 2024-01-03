#ifndef NORMALIZEDISPACH_H
#define NORMALIZEDISPACH_H

#include "tensor/tensor.h"

void normalize_cuda_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype);
void normalize_cpu_kernel(void* input, void* weight, void* bias, void* output, float eps, size_t lastshape, size_t headshape, size_t size, TENSORTYPE dtype);
// layernorm kernel


Tensor Tensor::normalize(Tensor& weight, Tensor& bias, size_t heads, float eps)
{
    Tensor result = Tensor(this->shape, this->dtype, this->device, this->device_id);

    auto lastshape = this->shape.back();
    auto headshape = lastshape / heads;

    if (this->device == DEVICE::CPU){

        normalize_cpu_kernel(this->data, weight.data, bias.data, result.data, eps, lastshape, headshape, this->get_element_count(), this->dtype);

    } else if (this->device == DEVICE::CUDA){
        
        normalize_cuda_kernel(this->data, weight.data, bias.data, result.data, eps, lastshape, headshape, this->get_element_count(), this->dtype);

    } else {
        throw std::runtime_error("Device not implemented");
    }

    return result;

}









#endif // NORMALIZEDISPACH_H