#ifndef SWISHMUL_HPP
#define SWISHMUL_HPP

#include "tensor/tensor.h"


void swishmul_cpu_kernel(void* input, void* other, void* output, int size, TENSORTYPE dtype);
void swishmul_cuda_kernel(void* input, void* other, void* output, int size, TENSORTYPE dtype);

inline Tensor Tensor::swishmul(Tensor& other){
    Tensor output = Tensor(this->shape, this->dtype, this->device, this->device_id);
    
    int size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        swishmul_cpu_kernel(this->data, other.data, output.data, size, dtype);
    }
    else{
        swishmul_cuda_kernel(this->data, other.data, output.data, size, dtype);
    }
    return output;
}







#endif  // SWISHMUL_HPP