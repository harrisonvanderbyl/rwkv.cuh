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
        Embedding(Tensor& weighta){
            this->weight = weighta;
        }
        Tensor operator()(std::vector<std::vector<size_t>> indices){

            if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < indices.size() * indices[0].size() || buffer.dtype != weight.dtype || buffer.device != weight.device){
                buffer = Tensor({indices.size(), indices[0].size(), weight.shape[1]}, weight.dtype, weight.device);
            }
            
            auto cbuf = buffer.cloneWithFalseReshape({indices.size(), indices[0].size(), weight.shape[1]});

            return weight.gather(indices, cbuf);
        }

        void cuda(){
            this->weight = this->weight.cuda();
        }
};