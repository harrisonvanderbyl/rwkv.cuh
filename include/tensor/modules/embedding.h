#include "tensor/tensor.h"
class Embedding
{
    public:
        Tensor weight;
        long max_batch;
        long max_seq;
        Embedding(){
        }
        Embedding(Tensor& weighta){
            this->weight = weighta;
        }
        Tensor operator()(std::vector<std::vector<size_t>> indices){

            return weight[indices];
        }

        void cuda(){
            this->weight = this->weight.cuda();
        }
};