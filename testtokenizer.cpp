
#include "tokenizer/tokenizer.hpp"
#include <iostream>
#include <cmath>


int main(){

    RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");

    for (size_t i = 1; i < pow(2, 16); i++){
        std::cout << i << " '" + worldTokenizer.decode({i}) + "'" << std::endl;
        std::cout.flush();
    }

}