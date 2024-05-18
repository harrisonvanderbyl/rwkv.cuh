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
            
            pool->debug(input[0], "ffn input k");
            pool->debug(input[1], "ffn input r");

            pool->sync();
            auto k = key(input[0]);

            check_for_errors();
            auto r = receptance(input[1]);

            pool->debug(k, "ffn k");
            pool->debug(r, "ffn r");

            check_for_errors();
            auto krs = k.relusquared();

            pool->debug(krs, "ffn krs");

            check_for_errors();
            pool->sync();
            auto v = this->value(krs); 

            pool->debug(v, "ffn v");
            
            check_for_errors();
            r.sigmoidmul(v, residual, output);

            pool->debug(output, "ffn output");

            check_for_errors();

        }

};