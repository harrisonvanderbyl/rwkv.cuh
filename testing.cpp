#include "tensor/tensor.h"
#include "tensor/modules/timeshift.h"

int main(){
    Tensor a = Tensor({1024,1024}, kUINT_8);
    Tensor aO = Tensor({1024}, kFLOAT_32);
    Tensor aR = Tensor({1024}, kFLOAT_32);
    for (size_t i = 0; i < a.shape[0]; i++){
        for (size_t j = 0; j < a.shape[1]; j++){
            a[i][j] = rand() % 255;
        }
        aO[i] = (rand() % 255)/255.0;
        aR[i] = (rand() % 255)/255.0;
    }

    Tensor b = Tensor({1, 4, 1024});
    for (size_t i = 0; i < b.shape[0]; i++){
        for (size_t j = 0; j < b.shape[1]; j++){
            for (size_t k = 0; k < b.shape[2]; k++){
                b[i][j][k] = ((i+1) * k)/1024.0 + 0.5;
            }
        }
    }

    Tensor c = Tensor({1, 4, 1024});
    Tensor c2 = Tensor({1, 4, 1024});

    auto threadpool = get_threadpool(64);


    std::cout << "Testing matmul" << std::endl;


    auto nextTensor = a.matmul(aO,aR,b,c);
    auto nextTensor2 = a.matmul(aO,aR,b,c2);

    threadpool->start();
    threadpool->sync(true);
    std::cout << "Finished testing matmul" << std::endl;
    std::cout << c << std::endl;
    std::cout << c2 << std::endl;


    // TimeShift timeshift = TimeShift(a);
    // auto b = a.gather({{0UL,1UL}});
    // auto newb = Tensor(b.shape);
    
    // auto starttime = std::chrono::high_resolution_clock::now();

    // for ( size_t i = 0; i < 256; i++){
    //     b.normalize(a[0],a[1],newb, 1);
    // }
    // std::cout << timeshift.state[0];
    // auto zm = timeshift(newb);



    // std::cout << threadpool->streams[0].jobs.size() << std::endl;


    // // threadpool->sync(true);
    // std::cout << "Input:" << b[0][1];
    // std::cout << "Input:" << newb[0][1];
    // std::cout << "Output:" << timeshift.state[0];
    // std::cout << "Expect something" << zm[0][1] << std::endl;
    // auto endtime = std::chrono::high_resolution_clock::now();

    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() << std::endl;



    


    // std::cout << newb[0][1] << std::endl;
    // std::cout << b << std::endl;
    // std::cout << a << std::endl;

    // std::cout << zm << std::endl;
    threadpool->stop();
}

//[0, 0.977977, ..., 20904.8, 20931.5]