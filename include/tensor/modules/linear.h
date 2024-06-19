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
    Tensor bias = Tensor();
    bool quantized = false;
    bool hasbias = false;
    bool splitHorizontal = true;
    size_t bmm_size = 1;
    size_t outshape;
    size_t inshape;
    MMACTFUNC activation = NONE;
    Linear()
    {
    }

    Linear(safetensors &model, std::string prefix, MMACTFUNC act = NONE)
    {

        this->activation = act;

        if (model.contains(prefix + ".weight.zero"))
        {
            this->range = model[prefix + ".weight.range"];
            this->offset = model[prefix + ".weight.zero"];
            this->weight = model[prefix + ".weight"];
            this->quantized = true;
        }
        else
        {
            this->weight = model[prefix + ".weight"];
        }

        this->outshape = weight.shape[0];
        this->inshape = weight.shape[1];

        if (this->weight.shape.size() == 3)
        {
            this->bmm_size = this->weight.shape[0];
            this->outshape = weight.shape[1];
            this->inshape = weight.shape[2];
        }

        if (model.contains(prefix + ".bias"))
        {
            this->bias = model[prefix + ".bias"].cloneWithFalseReshape({bmm_size,outshape});
            this->hasbias = true;
        }

        
    }

    // Copy constructor
    Linear(const Linear &other)
    {
        this->weight = other.weight;
        this->range = other.range;
        this->offset = other.offset;
        this->quantized = other.quantized;
        this->buffer = other.buffer;
        this->bias = other.bias;
        this->hasbias = other.hasbias;
        this->bmm_size = other.bmm_size;
        this->outshape = other.outshape;
        this->inshape = other.inshape;
        this->activation = other.activation;
    }

    // default copy assignment operator
    Linear &operator=(const Linear &other) = default;

    Tensor operator()(Tensor input)
    {

        if (buffer.data == nullptr || buffer.shape[1] * buffer.shape[2] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device)
        {
            buffer = *new Tensor({this->bmm_size, input.shape[0], input.shape[1], (outshape)}, input.dtype, input.device);
        }

        auto threadpool = get_threadpool();

        if (!this->hasbias)
        {
            buffer.empty();
        }
        else
        {
            for (size_t bb = 0; bb < bmm_size; bb += 1)
            {
                for (size_t i = 0; i < buffer.shape[1]; i++)
                {
                    for (size_t j = 0; j < buffer.shape[2]; j++)
                    {
                        buffer[bb][i][j].copyfrom(this->bias[bb]);
                    }
                }
            }
            //     threadpool->debug(this->bias,"time_decay");

            // threadpool->sync();
            //     threadpool->debug(this->buffer,"time_decay_buffer");
            // threadpool->sync();
        }

        if (this->quantized)
        {
            auto out = this->weight.matmul(this->range, this->offset, input, buffer, activation);
            if(bmm_size > 1){
                return out.cloneWithFalseReshape({bmm_size,input.shape[0], input.shape[1], outshape});    
            }
            return out.cloneWithFalseReshape({input.shape[0], input.shape[1], outshape});
        }
        else
        {
            auto out = this->weight.matmul(input, buffer, activation);
            if(bmm_size > 1){
                return out.cloneWithFalseReshape({bmm_size,input.shape[0], input.shape[1], outshape});    
            }
            return out.cloneWithFalseReshape({input.shape[0], input.shape[1], outshape});
        }
    }

    Tensor operator()(Tensor input, Tensor residual)
    {

        if (this->quantized)
        {
            return this->weight.matmul(this->range, this->offset, input, residual, activation);
        }
        else
        {
            return this->weight.matmul(input, residual, activation);
        }
    }

    void cuda()
    {
        this->weight = this->weight.cuda();

        if (this->quantized)
        {
            this->range = this->range.cuda();
            this->offset = this->offset.cuda();
        }
        if (this->hasbias)
        {
            this->bias = this->bias.cuda();
        }

        // this->buffer = this->buffer.cuda();
        // not needed, buffer will be recreated on device mismatch
    }
};

#endif