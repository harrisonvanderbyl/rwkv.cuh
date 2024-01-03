#ifndef LERP_HPP    
#define LERP_HPP

#include "tensor/tensor.h"

void lerp_cuda_kernel(void* w, void* A, void* B, void* output, size_t size, size_t loopsize, TENSORTYPE dtype);
void lerp_cpu_kernel(void* w, void* A, void* B, void* output, size_t size, size_t loopsize, TENSORTYPE dtype);


#ifndef __CUDACC__

void lerp_cuda_kernel(void* w, void* A, void* B, void* output, size_t size, size_t loopsize, TENSORTYPE dtype){
    throw std::runtime_error("Not compiled with cuda support");
}

#endif

Tensor Tensor::lerp(Tensor& A, Tensor& B){
    Tensor output = Tensor(A.shape, A.dtype, A.device, A.device_id);
    size_t loopsize = this->get_element_count();
    size_t size = A.get_element_count();

    auto dtype = this->dtype;

    if (this->device == DEVICE::CPU){
        lerp_cpu_kernel(this->data, A.data, B.data, output.data, size, loopsize, dtype);
    }
    else{
        lerp_cuda_kernel(this->data, A.data, B.data, output.data, size, loopsize, dtype);
    }  
    return output;
}


#endif // LERP_HPP