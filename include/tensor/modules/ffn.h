#include "tensor/tensor.h"

#include "tensor/safetensors.h"
#include "tensor/modules/timeshift.h"
#include "tensor/modules/linear.h"
class FeedForward
{
    public:
        uint32_t n_head; 
        Linear receptance;
        Linear key;
        Linear value;

        FeedForward(){
        }
        
        FeedForward(int layerID, safetensors& model){
            std::string prefix = "blocks." + std::to_string(layerID) + ".ffn.";

            

            this->receptance = Linear(model, prefix + "receptance", SIGMOIDMUL);
            this->key = Linear(model, prefix + "key", RELUSQUARE);
            this->value = Linear(model, prefix + "value",SETVALUE);
        }
        void operator()(Tensor input, Tensor residual, Tensor output){

            auto pool = get_threadpool();

            auto k = key(input[0]);
            pool->sync();
            auto v = value(k, input[0]); 
            auto r = receptance(input[1],residual);
        }

};