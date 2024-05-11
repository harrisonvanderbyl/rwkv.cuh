#ifndef LINEAR_HPP
#define LINEAR_HPP
#include "tensor/tensor.h"
#include "tensor/safetensors.h"
#include <iostream>
class Linear
{
    public:
        Tensor weight;
        Tensor range;
        Tensor offset;
        Tensor buffer = Tensor();
        bool quantized = false;
        bool splitHorizontal = true;
        Linear(){
            
        }

        Linear(safetensors& model, std::string prefix){
            if (model.contains(prefix + ".weight.zero")){
                this->range = model[prefix + ".weight.range"];
                this->offset = model[prefix + ".weight.zero"];
                this->weight = model[prefix + ".weight"];
                this->quantized = true;

            }else{
                this->weight = model[prefix + ".weight"];
            }
            
        }

        // Copy constructor
        Linear(const Linear& other){
            this->weight = other.weight;
            this->range = other.range;
            this->offset = other.offset;
            this->quantized = other.quantized;
            this->buffer = other.buffer;

        }
        
        // default copy assignment operator
        Linear& operator=(const Linear& other) = default;
    
        
        Tensor operator()(Tensor input) {

            if(buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({input.shape[0],input.shape[1], (weight.shape[0])}, input.dtype, input.device);
            }
            buffer.empty();


            if (this->quantized){
                return this->weight.matmul(this->range, this->offset, input, buffer).cloneWithFalseReshape({input.shape[0],input.shape[1], weight.shape[0]});
            }else{
                return this->weight.matmul(input, buffer).cloneWithFalseReshape({input.shape[0],input.shape[1], weight.shape[0]});
            }  
        }

        Tensor operator()(Tensor input, Tensor residual) {
          
            
            if (this->quantized){
                return this->weight.matmul(this->range, this->offset, input, residual);
            }else{
                return this->weight.matmul(input, residual);
            }  
        }

        void cuda(){
            this->weight = this->weight.cuda();
            if (this->quantized){
                this->range = this->range.cuda();
                this->offset = this->offset.cuda();
            }

            // this->buffer = this->buffer.cuda();
            // not needed, buffer will be recreated on device mismatch
        }

};

#endif