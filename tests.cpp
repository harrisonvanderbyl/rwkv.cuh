#include <iostream>
#include "tensor/safetensors.h"


int main(){
    

    Tensor bias = {{8}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}};
    Tensor weight = {{8}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}};
    Tensor input = {{2,8}, std::vector<float>{1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002}};

    auto output = input.normalize(weight,bias);

    std::cout << "Normalize tests" << std::endl;
    
    std::cout << output << std::endl;
    std::cout << output.get<float>(0) << "," << output.get<float>(1) << "," << output.get<float>(2) << "," << output.get<float>(3) << std::endl;

    std::cout << "Normalize tests bf16" << std::endl;

    Tensor bias_bf16 = {{8}, std::vector<bfloat16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    Tensor weight_bf16 = {{8}, std::vector<bfloat16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    Tensor input_bf16 = {{2,8}, std::vector<bfloat16>{1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002}};

    auto output_bf16 = input_bf16.normalize(weight_bf16,bias_bf16);

    std::cout << output_bf16 << std::endl;
    std::cout << (float)output_bf16.get<bfloat16>(0) << "," << (float)output_bf16.get<bfloat16>(1) << "," << (float)output_bf16.get<bfloat16>(2) << "," << (float)output_bf16.get<bfloat16>(3) << std::endl;

    // std::cout << "Normalize tests float cuda" << std::endl;

    // Tensor bias_cuda = {{8}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    // Tensor weight_cuda = {{8}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    // Tensor input_cuda = {{16}, std::vector<float>{1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002}};

    // // // bias_cuda = bias_cuda.cuda();
    // // // weight_cuda = weight_cuda.cuda();
    // // // input_cuda = input_cuda.cuda();

    // // // // auto output_cuda = input_cuda.normalize(weight_cuda,bias_cuda);

    // std::cout << output_cuda << std::endl;
    // // // // std::cout << output_cuda.get<float>(0) << "," << output_cuda.get<float>(1) << "," << output_cuda.get<float>(2) << "," << output_cuda.get<float>(3) << std::endl;

    // std::cout << "Normalize tests bf16 cuda" << std::endl;

    // Tensor bias_bf16_cuda = {{8}, std::vector<bfloat16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    // Tensor weight_bf16_cuda = {{8}, std::vector<bfloat16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0}};
    // Tensor input_bf16_cuda = {{16}, std::vector<bfloat16>{1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002}};
    // // // bias_bf16_cuda = bias_bf16_cuda.cuda();
    // // // weight_bf16_cuda = weight_bf16_cuda.cuda();
    // // // input_bf16_cuda = input_bf16_cuda.cuda();

    // // // // auto output_bf16_cuda = input_bf16_cuda.normalize(weight_bf16_cuda,bias_bf16_cuda);

    // std::cout << output_bf16_cuda << std::endl;
    // // // // std::cout << (float)output_bf16_cuda.get<bfloat16>(0) << "," << (float)output_bf16_cuda.get<bfloat16>(1) << "," << (float)output_bf16_cuda.get<bfloat16>(2) << "," << (float)output_bf16_cuda.get<bfloat16>(3) << std::endl;

    std::cout << "Normalize tests headsize 2" << std::endl;

    auto outputfloat = input.normalize(weight,bias,2);
    // // // // auto outputfloatcuda = input_cuda.normalize(weight_cuda,bias_cuda,2);

    std::cout << outputfloat << std::endl;
    // std::cout << outputfloatcuda << std::endl;

    std::cout << outputfloat.get<float>(0) << "," << outputfloat.get<float>(1) << "," << outputfloat.get<float>(2) << "," << outputfloat.get<float>(3) << std::endl;
    // // // // std::cout << outputfloatcuda.get<float>(0) << "," << outputfloatcuda.get<float>(1) << "," << outputfloatcuda.get<float>(2) << "," << outputfloatcuda.get<float>(3) << std::endl;

    std::cout << "Normalize tests headsize 2 bf16" << std::endl;

    auto outputbf16 = input_bf16.normalize(weight_bf16,bias_bf16,2);
    // // // // auto outputbf16cuda = input_bf16_cuda.normalize(weight_bf16_cuda,bias_bf16_cuda,2);

    std::cout << outputbf16 << std::endl;
    // std::cout << outputbf16cuda << std::endl;

    std::cout << (float)outputbf16.get<bfloat16>(0) << "," << (float)outputbf16.get<bfloat16>(1) << "," << (float)outputbf16.get<bfloat16>(2) << "," << (float)outputbf16.get<bfloat16>(3) << std::endl;
    // // // // std::cout << (float)outputbf16cuda.get<bfloat16>(0) << "," << (float)outputbf16cuda.get<bfloat16>(1) << "," << (float)outputbf16cuda.get<bfloat16>(2) << "," << (float)outputbf16cuda.get<bfloat16>(3) << std::endl;

    
   // relusquare test
   Tensor rtest = {{8}, std::vector<float>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
   // bfloat16 test
    Tensor rtest_bf16 = {{8}, std::vector<bfloat16>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    // cuda float test
    // Tensor rtest_cuda = {{8}, std::vector<float>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    // // // rtest_cuda = rtest_cuda.cuda();
    // cuda bfloat16 test
    // Tensor rtest_bf16_cuda = {{8}, std::vector<bfloat16>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    // // // rtest_bf16_cuda = rtest_bf16_cuda.cuda();

    auto rtest_out = rtest.relusquared();
    auto rtest_out_bf16 = rtest_bf16.relusquared();

    // // auto rtest_out_cuda = rtest_cuda.relusquared();
    // // auto rtest_out_bf16_cuda = rtest_bf16_cuda.relusquared();

    std::cout << "Relusquared tests" << std::endl;
    std::cout << rtest_out << std::endl;
    std::cout << rtest_out_bf16 << std::endl;
    // std::cout << rtest_out_cuda << std::endl;
    // std::cout << rtest_out_bf16_cuda << std::endl;

    std::cout << rtest_out.get<float>(0) << "," << rtest_out.get<float>(1) << "," << rtest_out.get<float>(2) << "," << rtest_out.get<float>(3) << std::endl;
    std::cout << (float)rtest_out_bf16.get<bfloat16>(0) << "," << (float)rtest_out_bf16.get<bfloat16>(1) << "," << (float)rtest_out_bf16.get<bfloat16>(2) << "," << (float)rtest_out_bf16.get<bfloat16>(3) << std::endl;
    // // // // std::cout << rtest_out_cuda.get<float>(0) << "," << rtest_out_cuda.get<float>(1) << "," << rtest_out_cuda.get<float>(2) << "," << rtest_out_cuda.get<float>(3) << std::endl;
    // // // // std::cout << (float)rtest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(3) << std::endl;

    // sigmoidmul test
    Tensor stest = {{8}, std::vector<float>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor stest2 = {{8}, std::vector<float>{0.8, 9.5, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};
    // bfloat16 test
    Tensor stest_bf16 = {{8}, std::vector<bfloat16>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor stest2_bf16 = {{8}, std::vector<bfloat16>{0.8, 9.5, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};
    // cuda float test
    // // Tensor stest_cuda = stest.cuda();
    // // Tensor stest2_cuda = stest2.cuda();
    // cuda bfloat16 test
    // // Tensor stest_bf16_cuda = stest_bf16.cuda();
    // // Tensor stest2_bf16_cuda = stest2_bf16.cuda();

    auto stest_out = stest.sigmoidmul(stest2);
    auto stest_out_bf16 = stest_bf16.sigmoidmul(stest2_bf16);

    // // // auto stest_out_cuda = stest_cuda.sigmoidmul(stest2_cuda);
    // // // auto stest_out_bf16_cuda = stest_bf16_cuda.sigmoidmul(stest2_bf16_cuda);

    std::cout << "Sigmoidmul tests" << std::endl;
    std::cout << stest_out << std::endl;
    std::cout << stest_out_bf16 << std::endl;
    // std::cout << stest_out_cuda << std::endl;
    // std::cout << stest_out_bf16_cuda << std::endl;

    std::cout << stest_out.get<float>(0) << "," << stest_out.get<float>(1) << "," << stest_out.get<float>(2) << "," << stest_out.get<float>(3) << std::endl;
    std::cout << (float)stest_out_bf16.get<bfloat16>(0) << "," << (float)stest_out_bf16.get<bfloat16>(1) << "," << (float)stest_out_bf16.get<bfloat16>(2) << "," << (float)stest_out_bf16.get<bfloat16>(3) << std::endl;
    // // // // std::cout << stest_out_cuda.get<float>(0) << "," << stest_out_cuda.get<float>(1) << "," << stest_out_cuda.get<float>(2) << "," << stest_out_cuda.get<float>(3) << std::endl;
    // // // // std::cout << (float)stest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)stest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)stest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)stest_out_bf16_cuda.get<bfloat16>(3) << std::endl;

    // lerp test
    Tensor ltest = {{8}, std::vector<float>{1.0, 1, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor ltest2 = {{8}, std::vector<float>{0.8, 2, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};
    Tensor lweight = {{2}, std::vector<float>{0.5, 2.0}};

    // bfloat16 test
    Tensor ltest_bf16 = {{8}, std::vector<bfloat16>{1.0, 1.0, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor ltest2_bf16 = {{8}, std::vector<bfloat16>{0.8, 2.0, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};
    Tensor lweight_bf16 = {{2}, std::vector<bfloat16>{0.5, 2.0}};

    // cuda float test
    // // Tensor ltest_cuda = ltest.cuda();
    // // Tensor ltest2_cuda = ltest2.cuda();
    // // Tensor lweight_cuda = lweight.cuda();

    // cuda bfloat16 test
    // // Tensor ltest_bf16_cuda = ltest_bf16.cuda();
    // // Tensor ltest2_bf16_cuda = ltest2_bf16.cuda();
    // // Tensor lweight_bf16_cuda = lweight_bf16.cuda();

    auto ltest_out = lweight.lerp(ltest,ltest2);
    auto ltest_out_bf16 = lweight_bf16.lerp(ltest_bf16,ltest2_bf16);

    // // // // auto ltest_out_cuda = lweight_cuda.lerp(ltest_cuda,ltest2_cuda);
    // // // // auto ltest_out_bf16_cuda = lweight_bf16_cuda.lerp(ltest_bf16_cuda,ltest2_bf16_cuda);

    std::cout << "Lerp tests" << std::endl;
    std::cout << ltest_out << std::endl;
    std::cout << ltest_out_bf16 << std::endl;
    // std::cout << ltest_out_cuda << std::endl;
    // std::cout << ltest_out_bf16_cuda << std::endl;

    std::cout << ltest_out.get<float>(0) << "," << ltest_out.get<float>(1) << "," << ltest_out.get<float>(2) << "," << ltest_out.get<float>(3) << std::endl;
    std::cout << (float)ltest_out_bf16.get<bfloat16>(0) << "," << (float)ltest_out_bf16.get<bfloat16>(1) << "," << (float)ltest_out_bf16.get<bfloat16>(2) << "," << (float)ltest_out_bf16.get<bfloat16>(3) << std::endl;
    // // // // std::cout << ltest_out_cuda.get<float>(0) << "," << ltest_out_cuda.get<float>(1) << "," << ltest_out_cuda.get<float>(2) << "," << ltest_out_cuda.get<float>(3) << std::endl;
    // // // // std::cout << (float)ltest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(3) << std::endl;
    
    // swishmul test
    Tensor wtest = {{8}, std::vector<float>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor wtest2 = {{8}, std::vector<float>{0.8, 9.5, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};

    // bfloat16 test
    Tensor wtest_bf16 = {{8}, std::vector<bfloat16>{1.0, -0.5, 2.0, -0.2, -0.1, -0.05, -0.03, -0.02}};
    Tensor wtest2_bf16 = {{8}, std::vector<bfloat16>{0.8, 9.5, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02}};

    // cuda float test
    // // Tensor wtest_cuda = wtest.cuda();
    // // Tensor wtest2_cuda = wtest2.cuda();

    // cuda bfloat16 test
    // // Tensor wtest_bf16_cuda = wtest_bf16.cuda();
    // // Tensor wtest2_bf16_cuda = wtest2_bf16.cuda();

    auto wtest_out = wtest.swishmul(wtest2);
    auto wtest_out_bf16 = wtest_bf16.swishmul(wtest2_bf16);

    // // // auto wtest_out_cuda = wtest_cuda.swishmul(wtest2_cuda);
    // // // auto wtest_out_bf16_cuda = wtest_bf16_cuda.swishmul(wtest2_bf16_cuda);

    std::cout << "Swishmul tests" << std::endl;
    std::cout << wtest_out << std::endl;
    std::cout << wtest_out_bf16 << std::endl;
    // std::cout << wtest_out_cuda << std::endl;
    // std::cout << wtest_out_bf16_cuda << std::endl;

    std::cout << wtest_out.get<float>(0) << "," << wtest_out.get<float>(1) << "," << wtest_out.get<float>(2) << "," << wtest_out.get<float>(3) << std::endl;
    std::cout << (float)wtest_out_bf16.get<bfloat16>(0) << "," << (float)wtest_out_bf16.get<bfloat16>(1) << "," << (float)wtest_out_bf16.get<bfloat16>(2) << "," << (float)wtest_out_bf16.get<bfloat16>(3) << std::endl;
    // // // // std::cout << wtest_out_cuda.get<float>(0) << "," << wtest_out_cuda.get<float>(1) << "," << wtest_out_cuda.get<float>(2) << "," << wtest_out_cuda.get<float>(3) << std::endl;
    // // // // std::cout << (float)wtest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(3) << std::endl;

    // wkvtest
    Tensor wkvtest = {{1,32,64,64}};
    Tensor wkvtest_bf16 = {{1,32,64,64}};
    
    for (int i = 0; i < 32*64*64; i++){
        ((float*)wkvtest.data)[i] = float(rand())/float(RAND_MAX);
    }
    wkvtest_bf16.copyfrom(wkvtest);
    Tensor k = {{1,1,32*64}};
    Tensor kbf16 = {{1,1,32*64}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 32*64; i++){
        ((float*)k.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)kbf16.data)[i] = bfloat16(((float*)k.data)[i]);
    }
    Tensor v = {{1,1,32*64}};
    Tensor vbf16 = {{1,1,32*64}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 32*64; i++){
        ((float*)v.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)vbf16.data)[i] = bfloat16(((float*)v.data)[i]);
    }
    Tensor r = {{1,1,32*64}};
    Tensor rbf16 = {{1,1,32*64}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 32*64; i++){
        ((float*)r.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)rbf16.data)[i] = bfloat16(((float*)r.data)[i]);
    }
    Tensor decay = {{32*64}};
    Tensor decaybf16 = {{32*64}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 32*64; i++){
        ((float*)decay.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)decaybf16.data)[i] = bfloat16(((float*)decay.data)[i]);
    }
    Tensor first = {{32*64}};
    Tensor firstbf16 = {{32*64}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 32*64; i++){
        ((float*)first.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)firstbf16.data)[i] = bfloat16(((float*)first.data)[i]);
    }

    // // Tensor wkvtest_cuda = wkvtest.cuda();
    

    auto out = wkvtest.wkv5(k,v,r,decay,first);

    std::cout << "wkvtest" << std::endl;
    std::cout << out << std::endl;
    std::cout << out.get<float>(0) << ","<< out.get<float>(1) <<","<< out.get<float>(2) << ","<<out.get<float>(3) << std::endl;
    std::cout << wkvtest.get<float>(0) << ","<<wkvtest.get<float>(1) <<","<< wkvtest.get<float>(2) <<","<< wkvtest.get<float>(3) << std::endl;

    // wkvtest cuda
    // // Tensor k_cuda = k.cuda();
    // // Tensor v_cuda = v.cuda();
    // // Tensor r_cuda = r.cuda();
    // // Tensor decay_cuda = decay.cuda();
    // // Tensor first_cuda = first.cuda();

    // // // // // // // auto out_cuda = wkvtest_cuda.wkv5(k_cuda,v_cuda,r_cuda,decay_cuda,first_cuda);

    // std::cout << "wkvtest cuda" << std::endl;
    // std::cout << out_cuda << std::endl;
    // // // // std::cout << out_cuda.get<float>(0) << ","<< out_cuda.get<float>(1) <<","<< out_cuda.get<float>(2) << ","<<out_cuda.get<float>(3) << std::endl;
    // // // // std::cout << wkvtest_cuda.get<float>(0) << ","<<wkvtest_cuda.get<float>(1) <<","<< wkvtest_cuda.get<float>(2) <<","<< wkvtest_cuda.get<float>(3) << std::endl;

    // // auto wkvtest_bf16_cuda = wkvtest_bf16.cuda();
    // wkvtest bf16
    auto out_bf16 = wkvtest_bf16.wkv5(kbf16,vbf16,rbf16,decaybf16,firstbf16);

    std::cout << "wkvtest bf16" << std::endl;
    std::cout << out_bf16 << std::endl;
    std::cout << (float)out_bf16.get<bfloat16>(0) << ","<< (float)out_bf16.get<bfloat16>(1) <<","<< (float)out_bf16.get<bfloat16>(2) << ","<<(float)out_bf16.get<bfloat16>(3) << std::endl;

    // wkvtest cuda bf16
    // // auto kbf16_cuda = kbf16.cuda();
    // // auto vbf16_cuda = vbf16.cuda();
    // // auto rbf16_cuda = rbf16.cuda();
    // // auto decaybf16_cuda = decaybf16.cuda();
    // // auto firstbf16_cuda = firstbf16.cuda();

    // // // // // // // auto out_bf16_cuda = wkvtest_bf16_cuda.wkv5(kbf16_cuda,vbf16_cuda,rbf16_cuda,decaybf16_cuda,firstbf16_cuda);

    // std::cout << "wkvtest cuda bf16" << std::endl;
    // std::cout << out_bf16_cuda << std::endl;
    // // // // std::cout << (float)out_bf16_cuda.get<bfloat16>(0) << ","<< (float)out_bf16_cuda.get<bfloat16>(1) <<","<< (float)out_bf16_cuda.get<bfloat16>(2) << ","<<(float)out_bf16_cuda.get<bfloat16>(3) << std::endl;
    // // // // std::cout << (float)wkvtest_bf16_cuda.get<float>(0) << ","<< (float)wkvtest_bf16_cuda.get<float>(1) <<","<< (float)wkvtest_bf16_cuda.get<float>(2) << ","<<(float)wkvtest_bf16_cuda.get<float>(3) << std::endl;

    // matmultest 
    std::cout << "matmultest" << std::endl;
    Tensor matmultest = {{256,256}};
    Tensor matmultest_bf16 = {{256,256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256*256; i++){
        ((float*)matmultest.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)matmultest_bf16.data)[i] = bfloat16(((float*)matmultest.data)[i]);
    }

    Tensor matmultest2 = {{1,256,256}};
    Tensor matmultest2_bf16 = {{1,256,256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256*256; i++){
        ((float*)matmultest2.data)[i] = float(rand())/float(RAND_MAX);
        ((bfloat16*)matmultest2_bf16.data)[i] = bfloat16(((float*)matmultest2.data)[i]);
    }

    std::cout << "Matmultestprematmul" << std::endl;

    auto outputmm = matmultest.matmul(matmultest2);

    std::cout << "Matmultestprematmul2" << std::endl;
    auto outputmm_bf16 = matmultest_bf16.matmul(matmultest2_bf16);

    std::cout << "matmultest2" << std::endl;
    std::cout << outputmm << std::endl;
    std::cout << outputmm_bf16 << std::endl;
    std::cout << outputmm.get<float>(0) << ","<< outputmm.get<float>(1) <<","<< outputmm.get<float>(2) << ","<<outputmm.get<float>(3) << std::endl;
    std::cout << (float)outputmm_bf16.get<bfloat16>(0) << ","<< (float)outputmm_bf16.get<bfloat16>(1) <<","<< (float)outputmm_bf16.get<bfloat16>(2) << ","<<(float)outputmm_bf16.get<bfloat16>(3) << std::endl;

    // matmultest cuda
    // std::cout << "matmultest cuda" << std::endl;
    // // Tensor matmultest_cuda = matmultest.cuda();
    // // Tensor matmultest2_cuda = matmultest2.cuda();

    // std::cout << "Matmultestprematmul cuda" << std::endl;

    // // // auto outputmm_cuda = matmultest_cuda.matmul(matmultest2_cuda);
    
    // std::cout << "matmultest bf16 cuda" << std::endl;

    // // Tensor matmultest_bf16_cuda = matmultest_bf16.cuda();
    // // Tensor matmultest2_bf16_cuda = matmultest2_bf16.cuda();

    // std::cout << "Matmultestprematmul bf16 cuda" << std::endl;

    // // // auto outputmm_bf16_cuda = matmultest_bf16_cuda.matmul(matmultest2_bf16_cuda);

    // std::cout << "matmultest2 cuda" << std::endl;
    // std::cout << outputmm_cuda << std::endl;
    // std::cout << outputmm_bf16_cuda << std::endl;
    // // // // std::cout << outputmm_cuda.get<float>(0) << ","<< outputmm_cuda.get<float>(1) <<","<< outputmm_cuda.get<float>(2) << ","<<outputmm_cuda.get<float>(3) << std::endl;
    // // // // std::cout << (float)outputmm_bf16_cuda.get<bfloat16>(0) << ","<< (float)outputmm_bf16_cuda.get<bfloat16>(1) <<","<< (float)outputmm_bf16_cuda.get<bfloat16>(2) << ","<<(float)outputmm_bf16_cuda.get<bfloat16>(3) << std::endl;

    // matmultest cuda




    return 0;
}