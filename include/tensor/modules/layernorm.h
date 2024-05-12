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
        Tensor buffer = Tensor();
        
        LayerNorm(Tensor weighti, Tensor biasi, size_t heads = 1, float eps = 1e-5){
            this->weight = weighti;
            this->bias = biasi;
            this->heads = heads;
            this->eps = eps;

        }

        // default copy assignment operator
        LayerNorm& operator=(const LayerNorm& other) = default;

        LayerNorm(){
        }

        // copy constructor
        LayerNorm(const LayerNorm& other){
            this->weight = other.weight;
            this->bias = other.bias;
            this->heads = other.heads;
            this->eps = other.eps;
            this->buffer = other.buffer;
        }

        Tensor operator()(Tensor input){

            if(buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }
            
            this->buffer.empty();

            return input.normalize(this->weight, this->bias, buffer, this->heads, this->eps).cloneWithFalseReshape({input.shape[0],input.shape[1], input.shape[2]});
        }

        void cuda(){
            this->weight = this->weight.cuda();
            this->bias = this->bias.cuda();
            // this->buffer = this->buffer.cuda();
            // this isnt necessary because the buffer will reallocate on mismatch
        }

};



#endif