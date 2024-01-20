#ifndef LERP_HPP
#define LERP_HPP

#include "tensor/tensor.h"

void lerp_cuda_kernel(void *w, void *A, void *B, void *output, size_t size, size_t loopsize, TENSORTYPE dtype);
void lerp_cpu_kernel(void *w, void *A, void *B, void *output, size_t size, size_t loopsize, TENSORTYPE dtype);

inline Tensor Tensor::lerp(Tensor &A, Tensor &B, Tensor &output, bool v6)
{
    size_t loopsize = this->get_element_count();
    size_t size = A.get_element_count();

    if (v6)
    {

        if (this->device == DEVICE::CPU)
        {
            lerp_cpu_kernel(this->data, B.data, A.data, output.data, size, loopsize, dtype);
        }
        else
            CUDAONLY
            {
                lerp_cuda_kernel(this->data, B.data, A.data, output.data, size, loopsize, dtype);
            }
    }
    else
    {
        if (this->device == DEVICE::CPU)
        {
            lerp_cpu_kernel(this->data, A.data, B.data, output.data, size, loopsize, dtype);
        }
        else
            CUDAONLY
            {
                lerp_cuda_kernel(this->data, A.data, B.data, output.data, size, loopsize, dtype);
            }
    }
    return output;
}

#endif // LERP_HPP