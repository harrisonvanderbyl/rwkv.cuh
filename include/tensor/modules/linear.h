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
        Tensor buffer;
        bool quantized = false;
        
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

            
        }
        
    
        
        Tensor operator()(Tensor& input) {

            if (this->quantized){
                return this->weight.matmul(this->range, this->offset, input);
            }else{
                return this->weight.matmul(input);
            }  
        }

        Tensor operator()(Tensor& input, Tensor& residual) {

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
        }

};

#endif