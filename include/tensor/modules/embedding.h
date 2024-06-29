#include "tensor/tensor.h"
class Embedding
{
    public:
        Tensor weight;
        long max_batch;
        long max_seq;
        Tensor buffer;
        Embedding(){
        }
        Embedding(Tensor weighta){
            this->weight = weighta;
        }
        Tensor operator()(std::vector<std::vector<size_t>> indices){

            if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < indices.size() * indices[0].size() || buffer.dtype != weight.dtype ){
                buffer = Tensor({indices.size(), indices[0].size(), weight.shape[1]}, weight.dtype, buffer.device);
            }
            auto cbuf = buffer.cloneWithFalseReshape({indices.size(), indices[0].size(), weight.shape[1]});
            if(indices.size() == 1 && indices[0].size() == 1){
                return weight[indices[0][0]].cloneWithFalseReshape({1, 1, weight.shape[1]});
            }
            return weight.gather(indices, cbuf);
        }
        Tensor operator()(){

            if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < weight.shape[0] * weight.shape[1] || buffer.dtype != weight.dtype ){
                buffer = Tensor({1, weight.shape[0] , weight.shape[1]}, weight.dtype, buffer.device);
            }
            buffer.copyfrom(weight);
            return buffer;
        }

        void cuda(){
            this->buffer = this->buffer.cuda();
            this->weight = this->weight.cuda();
        }
};