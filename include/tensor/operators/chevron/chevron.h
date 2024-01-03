#ifndef CHEVRON_H
#define CHEVRON_H

#include "tensor/tensor.h"

// chevron for std::cout
static std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    std::string shapestring = "(";
    for (int i = 0; i < t.shape.size(); i++) {
        shapestring += std::to_string(t.shape[i]);
        if (i != t.shape.size() - 1) {
            shapestring += ", ";
        }
    }
    shapestring += ")";


    os << "Tensor(shape=" << shapestring << ", dtype=" << get_dtype_name(t.dtype) << ", device=" << get_device_name(t.device) << ")";
    if (t.dtype == TENSORTYPE::kFLOAT_32){
        os << "\n";
        os << "[";
        os << t.get<float>(0) << ", ";
        if(t.get_element_count() > 1)
            os << t.get<float>(1) << ", ";
        if(t.get_element_count() > 2)
            os << "...";
        if(t.get_element_count() > 3){
            os << ", " << t.get<float>(t.get_element_count() - 2);
            os << ", " << t.get<float>(t.get_element_count() - 1);
        }
        

        os << "]\n";
    }
    if (t.dtype == TENSORTYPE::kBFLOAT_16){
        os << "\n";
        os << "[";
        os << float(t.get<bfloat16>(0)) << ", ";
        if(t.get_element_count() > 1)
            os << float(t.get<bfloat16>(1)) << ", ";
        if(t.get_element_count() > 2)
            os << "...";
        if(t.get_element_count() > 3){
            os << ", " << float(t.get<bfloat16>(t.get_element_count() - 2));
            os << ", " << float(t.get<bfloat16>(t.get_element_count() - 1));
        }
        

        os << "]\n";
    }
    return os;
}

#endif