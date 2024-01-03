#ifndef GATHER_H
#define GATHER_H
#include "tensor/tensor.h"
Tensor Tensor::operator[](const size_t index) {
    std::vector<size_t> new_shape;
    size_t skipdata = get_dtype_bytes(this->dtype);
    for (int i = 1; i < shape.size(); i++) {
        new_shape.push_back(shape[i]);
        skipdata *= shape[i];
    }
    void* ndata = (void*)((char*)this->data + index * skipdata);
    
    auto out = Tensor(new_shape, ndata, this->dtype, this->device, this->device_id);

    return out;
}

Tensor Tensor::operator[](std::vector<std::vector<size_t>> indices) {
    
    std::vector<size_t> new_shape = {indices.size(), indices[0].size(), this->shape[1]};
    auto out = Tensor(new_shape, this->dtype, this->device, this->device_id);

    for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < indices[0].size(); j++) {
            if (indices[i][j] >= this->shape[0]) {
                throw std::invalid_argument("Index out of range");
            }
            
            if (this->device == DEVICE::CPU) {
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                memcpy(to, from, this->shape[1] * get_dtype_bytes(this->dtype));
            } else {
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                cudaMemcpy(to, from, this->shape[1] * get_dtype_bytes(this->dtype), cudaMemcpyDeviceToDevice);
            }
        }
    }

    return out;
    
}







#endif