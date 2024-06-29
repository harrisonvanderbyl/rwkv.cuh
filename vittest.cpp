#include "png/vit_image_preprocessor.hpp"
#include "tokenizer/tokenizer.hpp"
#include "sampler/sample.h"
/*
{'crop_size': 336, 'do_center_crop': True, 'do_normalize': True, 'do_resize': True, 'feature_extractor_type': 'CLIPFeatureExtractor', 'image_mean': [0.48145466, 0.4578275, 0.40821073], 'image_std': [0.26862954, 0.26130258, 0.27577711], 'resample': 3, 'size': 336}
*/

RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
#include "iostream"
int main(int argsc, char **args)
{
    auto pngfile = "./image.png";
    RWKV model("./model.safetensors", 8);
    
    ImageProcessor Ip(model.model);
    auto tensor = Ip.process_image(pngfile);
    get_threadpool()->sync(true);
    //  tensor([ 0.0155,  0.2856, -0.1434,  ...,  0.0176, -0.3926, -0.2852]
    auto tokens = worldTokenizer.encode("What is this image of?\n\nAssistant:");
    auto probs = model({tokens}, tensor);
    probs = probs[0][probs.shape[1]-1];
    get_threadpool()->sync(true);
    std::cout << (probs);
    // lodepng::save_file(buffer,"./testout.png");
    for(int z = 0; z < 100; z++){
       auto output = dart((float*)probs.data,1.0);
       std::cout << worldTokenizer.decode({output});
       probs = model({{output}});
       get_threadpool()->sync(true);
    }
}

/*
torch.Size([1, 577, 1024]) tensor([ 0.0155,  0.2856, -0.1434,  ...,  0.0176, -0.3926, -0.2852],
       grad_fn=<SelectBackward0>)
torch.Size([1, 577, 1024]) tensor([ 0.0900, -0.1784,  0.2354,  ...,  0.0749, -0.3618, -0.1639],
       grad_fn=<SelectBackward0>)
torch.Size([1, 577, 1024]) tensor([ 0.0804, -0.1736,  0.2136,  ...,  0.0715, -0.3604, -0.1664],
       grad_fn=<SelectBackward0>)
*/