#ifndef RESHAPE_H
#define RESHAPE_H

#include "tensor/tensor.h"

inline Tensor Tensor::reshape(std::vector<size_t> inshape) {
    Tensor ret = *this;
    ret.shape = shape.slice(1);
    ret.data_size_in_bytes = get_dtype_bytes(dtype);
    for (size_t i = 0; i < inshape.size(); i++){
        ret.data_size_in_bytes *= inshape[i];
    }
    assert(ret.data_size_in_bytes == data_size_in_bytes);\
    

    return ret;
}

#endif // RESHAPE_H