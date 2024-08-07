#ifndef GATHER_H
#define GATHER_H
#include "tensor/tensor.h"
inline Tensor Tensor::operator[](const size_t index) {
    auto newshape = shape.slice(1);
    size_t skipdata = get_dtype_bytes(dtype);
    for (size_t i = 1; i < shape.size(); i++) {
        skipdata *= shape[i];
    }
    void* ndata = (void*)((char*)data + index * skipdata);
    
    auto out = Tensor(newshape, ndata, dtype, device, device_id);

    return out;
}

inline Tensor Tensor::gather(std::vector<std::vector<size_t>> indices, Tensor out = Tensor()) {
    
    if (out.data == nullptr) {
        std::cout << "out.data is null\n";
        out = Tensor({indices.size(), indices[0].size(), shape[1]}, dtype, out.device, device_id);
    }

    for (size_t i = 0; i < indices.size(); i++) {
        for (size_t j = 0; j < indices[0].size(); j++) {
            if (indices[i][j] >= shape[0]) {
                throw std::invalid_argument("Index out of range");
            }
            
            if (device == DEVICE::CPU && out.device == DEVICE::CPU) {
                auto to = out[i][j].data;
                auto from = (*this)[indices[i][j]].data;
                memcpy(to, from, shape[1] * get_dtype_bytes(dtype));
            } else if (device == DEVICE::CPU && out.device == DEVICE::CUDA)
            {
                RcudaMemcpy(out[i][j].data, (*this)[indices[i][j]].data, shape[1] * get_dtype_bytes(dtype), cudaMemcpyHostToDevice);
            }else{
                RcudaMemcpy(out[i][j].data, (*this)[indices[i][j]].data, shape[1] * get_dtype_bytes(dtype), cudaMemcpyDeviceToDevice);
            }
        }
    }

    return out;
    
}







#endif