#include "tensor/tensor.h"
#include "tensor/modules/timeshift.h"

int main( int argc, char* argv[] ){
    Tensor a = Tensor({5,2048});
    Tensor b = Tensor({2048});
    Tensor c = Tensor({2048});
    Tensor d = Tensor({5,2048});

    ThreadPool* threadpool = get_threadpool(argc > 1 ? std::stoi(argv[1]) : 0);
    for (int i = 0; i < 2048; i++){
        for (int j = 0; j < 5; j++){
            a[j][i] = rand() % 100;
        }
    }
    for (int i = 0; i < 2048; i++){
        b[i] = rand() % 100;
    }
    for (int i = 0; i < 2048; i++){
        c[i] = rand() % 100;
    }

    a.normalize(b,c,d, argc > 2 ? std::stoi(argv[2]) : 1);

    threadpool->sync();

    a.lerp(b,c,d);

    threadpool->start();



    threadpool->sync(true);

    std::cout << d << std::endl;
}

//[0, 0.977977, ..., 20904.8, 20931.5]