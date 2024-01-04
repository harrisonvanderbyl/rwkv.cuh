#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP
#include "tensor/tensor.h"
class LayerNorm
{
    public:
        Tensor weight;
        Tensor bias;
        size_t heads = 1;
        float eps;
        
        LayerNorm(Tensor weighti, Tensor biasi, size_t heads = 1, float eps = 1e-5){
            this->weight = weighti;
            this->bias = biasi;
            this->heads = heads;
            this->eps = eps;

        }

        LayerNorm(){
        }

        // copy constructor
        LayerNorm(const LayerNorm& other){
            this->weight = other.weight;
            this->bias = other.bias;
            this->heads = other.heads;
            this->eps = other.eps;
        }

        Tensor operator()(Tensor& input){
            
            return input.normalize(this->weight, this->bias, this->heads, this->eps);
        }

        void cuda(){
            this->weight = this->weight.cuda();
            this->bias = this->bias.cuda();
        }

};



#endif