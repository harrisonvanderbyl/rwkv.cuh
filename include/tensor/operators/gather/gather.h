#ifndef GATHER_H
#define GATHER_H
#include "tensor/tensor.h"
inline Tensor Tensor::operator[](const size_t index) {
    std::vector<size_t> new_shape;
    size_t skipdata = get_dtype_bytes(dtype);
    for (int i = 1; i < shape.size(); i++) {
        new_shape.push_back(shape[i]);
        skipdata *= shape[i];
    }
    void* ndata = (void*)((char*)data + index * skipdata);
    
    auto out = Tensor(new_shape, ndata, dtype, device, device_id);

    return out;
}

inline Tensor Tensor::gather(std::vector<std::vector<size_t>> indices, Tensor out) {
    
    if (out.data == nullptr) {
        out = Tensor({indices.size(), indices[0].size(), shape[1]}, dtype, out.device, device_id);
    }

    for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < indices[0].size(); j++) {
            if (indices[i][j] >= shape[0]) {
                throw std::invalid_argument("Index out of range");
            }
            
            if (device == DEVICE::CPU && out.device == DEVICE::CPU) {
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                memcpy(to, from, shape[1] * get_dtype_bytes(dtype));
            } else if (device == DEVICE::CPU && out.device == DEVICE::CUDA)
            {
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                cudaMemcpy(to, from, shape[1] * get_dtype_bytes(dtype), cudaMemcpyDeviceToDevice);
            }else{
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                cudaMemcpy(to, from, shape[1] * get_dtype_bytes(dtype), cudaMemcpyHostToDevice);
            }
        }
    }

    return out;
    
}







#endif