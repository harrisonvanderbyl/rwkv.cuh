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

            

            this->receptance = Linear(model, prefix + "receptance");
            this->key = Linear(model, prefix + "key");
            this->value = Linear(model, prefix + "value");
        }
        void operator()(Tensor input, Tensor residual, Tensor output){

            auto pool = get_threadpool();
            
            pool->sync();
            auto k = key(input[0]);
            auto r = receptance(input[1]);

            auto krs = k.relusquared();

            pool->sync();
            auto v = this->value(krs); 
            
            r.sigmoidmul(v, residual, output);


        }

};