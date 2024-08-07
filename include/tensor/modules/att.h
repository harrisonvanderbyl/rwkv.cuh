#include "tensor/tensor.h"

#include "tensor/safetensors.h"
#include "tensor/modules/timeshift.h"
#include "tensor/modules/linear.h"
#include "tensor/modules/layernorm.h"
class Attention
{
public:
    uint32_t head_size = 64;
    uint32_t n_head;
    Tensor time_faaaa;
    Tensor state;
    Linear receptance;
    Linear key;
    Linear value;
    Linear gate;
    Linear output;
    LayerNorm ln_x;
    Tensor buffer;
    Linear w1;
    Linear w2;
    int layer = 0;

    Attention()
    {
    }

    Attention(int layerID, safetensors &model, size_t batch_size = 1)
    {
        // std::cout << "Attentioncreate:" << layerID << std::endl;
        std::string prefix = "blocks." + std::to_string(layerID) + ".att.";
        this->layer = layerID;

        auto dims = model[prefix + "receptance.weight"].shape[1];
        this->n_head = dims / this->head_size;
        this->state = Tensor({batch_size, this->n_head, this->head_size, this->head_size});

        this->time_faaaa = model[prefix + "time_faaaa"];

        this->receptance = Linear(model, prefix + "receptance");
        this->key = Linear(model, prefix + "key");
        this->value = Linear(model, prefix + "value");
        this->gate = Linear(model, prefix + "gate", SWISHMUL);
        this->output = Linear(model, prefix + "output");
        this->ln_x = LayerNorm(model[prefix + "ln_x.weight"], model[prefix + "ln_x.bias"], n_head, 64e-5);

        this->w1 = Linear(model, prefix + "w1", TANH);
        this->w2 = Linear(model, prefix + "w2", EXPNEGEXP);
    }

    Tensor operator()(Tensor input, Tensor &residual)
    {

        auto pool = get_threadpool();
        pool->sync();
        if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] * buffer.shape[2] < input.shape[1] * input.shape[2] * input.shape[3] || buffer.dtype != input.dtype || buffer.device != input.device)
        {
            buffer = *new Tensor(input.shape.slice(1), input.dtype, input.device);
        }

        auto cbuf = buffer.cloneWithFalseReshape({input.shape[1], input.shape[2], input.shape[3]});

        
        auto ww = this->w1(input[0]);

        pool->sync();

        auto w = this->w2(ww);
        auto k = this->key(input[1]);
        auto r = this->receptance(input[3]);
        auto v = this->value(input[2]);

        auto xm = this->state.wkv5(r, k, v, w, this->time_faaaa, cbuf);
        
        auto xxa = this->ln_x(xm);
        auto gv = this->gate(input[4], xxa);
        
        pool->sync();
        auto out = this->output(gv, residual);
        

        return out;
    }
};