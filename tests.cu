#include <iostream>
#include "tensor/safetensors.h"


int main(){
    
    Tensor bias = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)bias.data)[i] = float(rand())/float(RAND_MAX);
    }
    Tensor weight = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)weight.data)[i] = float(rand())/float(RAND_MAX);
    }
    Tensor input = {{2,256}};
    for (int i = 0; i < 2*256; i++){
        ((float*)input.data)[i] = float(rand())/float(RAND_MAX);
    }

    auto output = input.normalize(weight,bias);

    std::cout << "Normalize tests" << std::endl;
    
    std::cout << output << std::endl;
    std::cout << output.get<float>(0) << "," << output.get<float>(1) << "," << output.get<float>(2) << "," << output.get<float>(3) << std::endl;

    std::cout << "Normalize tests bf16" << std::endl;

    Tensor bias_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256; i++){
        ((bfloat16*)bias_bf16.data)[i] = bfloat16(((float*)bias.data)[i]);
    }
    Tensor weight_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256; i++){
        ((bfloat16*)weight_bf16.data)[i] = bfloat16(((float*)weight.data)[i]);
    }
    Tensor input_bf16 = {{2,256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 2*256; i++){
        ((bfloat16*)input_bf16.data)[i] = bfloat16(((float*)input.data)[i]);
    }

    auto output_bf16 = input_bf16.normalize(weight_bf16,bias_bf16);

    std::cout << output_bf16 << std::endl;
    std::cout << (float)output_bf16.get<bfloat16>(0) << "," << (float)output_bf16.get<bfloat16>(1) << "," << (float)output_bf16.get<bfloat16>(2) << "," << (float)output_bf16.get<bfloat16>(3) << std::endl;

    std::cout << "Normalize tests float cuda" << std::endl;

    Tensor bias_cuda = bias.cuda();
    Tensor weight_cuda = weight.cuda();
    Tensor input_cuda = input.cuda();

    auto output_cuda = input_cuda.normalize(weight_cuda,bias_cuda);

    std::cout << output_cuda << std::endl;
    std::cout << output_cuda.get<float>(0) << "," << output_cuda.get<float>(1) << "," << output_cuda.get<float>(2) << "," << output_cuda.get<float>(3) << std::endl;

    std::cout << "Normalize tests bf16 cuda" << std::endl;

    Tensor bias_bf16_cuda = bias_bf16.cuda();
    Tensor weight_bf16_cuda = weight_bf16.cuda();
    Tensor input_bf16_cuda = input_bf16.cuda();
 

    auto output_bf16_cuda = input_bf16_cuda.normalize(weight_bf16_cuda,bias_bf16_cuda);

    std::cout << output_bf16_cuda << std::endl;
    std::cout << (float)output_bf16_cuda.get<bfloat16>(0) << "," << (float)output_bf16_cuda.get<bfloat16>(1) << "," << (float)output_bf16_cuda.get<bfloat16>(2) << "," << (float)output_bf16_cuda.get<bfloat16>(3) << std::endl;

    std::cout << "Normalize tests headsize 2" << std::endl;

    auto outputfloat = input.normalize(weight,bias,2);
    auto outputfloatcuda = input_cuda.normalize(weight_cuda,bias_cuda,2);

    std::cout << outputfloat << std::endl;
    std::cout << outputfloatcuda << std::endl;

    std::cout << outputfloat.get<float>(0) << "," << outputfloat.get<float>(1) << "," << outputfloat.get<float>(2) << "," << outputfloat.get<float>(3) << std::endl;
    std::cout << outputfloatcuda.get<float>(0) << "," << outputfloatcuda.get<float>(1) << "," << outputfloatcuda.get<float>(2) << "," << outputfloatcuda.get<float>(3) << std::endl;

    std::cout << "Normalize tests headsize 2 bf16" << std::endl;

    auto outputbf16 = input_bf16.normalize(weight_bf16,bias_bf16,2);
    auto outputbf16cuda = input_bf16_cuda.normalize(weight_bf16_cuda,bias_bf16_cuda,2);

    std::cout << outputbf16 << std::endl;
    std::cout << outputbf16cuda << std::endl;

    std::cout << (float)outputbf16.get<bfloat16>(0) << "," << (float)outputbf16.get<bfloat16>(1) << "," << (float)outputbf16.get<bfloat16>(2) << "," << (float)outputbf16.get<bfloat16>(3) << std::endl;
    std::cout << (float)outputbf16cuda.get<bfloat16>(0) << "," << (float)outputbf16cuda.get<bfloat16>(1) << "," << (float)outputbf16cuda.get<bfloat16>(2) << "," << (float)outputbf16cuda.get<bfloat16>(3) << std::endl;

    
   // relusquare test
    std::cout << "Relusquared tests start" << std::endl;
   Tensor rtest = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)rtest.data)[i] = float(rand())/float(RAND_MAX) - 0.5;
    }
   // bfloat16 test
    Tensor rtest_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};

    for (int i = 0; i < 256; i++){
        ((bfloat16*)rtest_bf16.data)[i] = bfloat16(((float*)rtest.data)[i]);
    }
    // cuda float test
    Tensor rtest_cuda = rtest.cuda();
    // cuda bfloat16 test
    Tensor rtest_bf16_cuda = rtest_bf16.cuda();

    std::cout << "Relusquared tests starting" << std::endl;

    auto rtest_out = rtest.relusquared();
    auto rtest_out_bf16 = rtest_bf16.relusquared();

    std::cout << "Relusquared tests starting cuda" << std::endl;

    auto rtest_out_cuda = rtest_cuda.relusquared();
    auto rtest_out_bf16_cuda = rtest_bf16_cuda.relusquared();

    std::cout << "Relusquared tests" << std::endl;
    std::cout << rtest_out << std::endl;
    std::cout << rtest_out_bf16 << std::endl;
    std::cout << rtest_out_cuda << std::endl;
    std::cout << rtest_out_bf16_cuda << std::endl;

    std::cout << rtest_out.get<float>(0) << "," << rtest_out.get<float>(1) << "," << rtest_out.get<float>(2) << "," << rtest_out.get<float>(3) << std::endl;
    std::cout << (float)rtest_out_bf16.get<bfloat16>(0) << "," << (float)rtest_out_bf16.get<bfloat16>(1) << "," << (float)rtest_out_bf16.get<bfloat16>(2) << "," << (float)rtest_out_bf16.get<bfloat16>(3) << std::endl;
    std::cout << rtest_out_cuda.get<float>(0) << "," << rtest_out_cuda.get<float>(1) << "," << rtest_out_cuda.get<float>(2) << "," << rtest_out_cuda.get<float>(3) << std::endl;
    std::cout << (float)rtest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)rtest_out_bf16_cuda.get<bfloat16>(3) << std::endl;

    std::cout << "Relusquared tests end" << std::endl;

    // sigmoid test
    std::cout << "Sigmoid tests start" << std::endl;
    // sigmoidmul test
    Tensor stest = rtest;
    Tensor stest2 = rtest;
    Tensor stest3 = rtest;
    // bfloat16 test
    Tensor stest_bf16 = rtest_bf16;
    Tensor stest2_bf16 = rtest_bf16;
    Tensor stest3_bf16 = rtest_bf16;
    // cuda float test
    Tensor stest_cuda = stest.cuda();
    Tensor stest2_cuda = stest2.cuda();
    Tensor stest3_cuda = stest3.cuda();
    // cuda bfloat16 test
    Tensor stest_bf16_cuda = stest_bf16.cuda();
    Tensor stest2_bf16_cuda = stest2_bf16.cuda();
    Tensor stest3_bf16_cuda = stest3_bf16.cuda();

    std::cout << "Sigmoid tests starting" << std::endl;

    auto stest_out = stest.sigmoidmul(stest2, stest3);
    auto stest_out_bf16 = stest_bf16.sigmoidmul(stest2_bf16, stest3_bf16);

    std::cout << "Sigmoid tests starting cuda" << std::endl;

    auto stest_out_cuda = stest_cuda.sigmoidmul(stest2_cuda, stest3_cuda);
    auto stest_out_bf16_cuda = stest_bf16_cuda.sigmoidmul(stest2_bf16_cuda, stest3_bf16_cuda);

    std::cout << "Sigmoidmul tests" << std::endl;
    std::cout << stest_out << std::endl;
    std::cout << stest_out_bf16 << std::endl;
    std::cout << stest_out_cuda << std::endl;
    std::cout << stest_out_bf16_cuda << std::endl;

    // lerp test
    Tensor ltest = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)ltest.data)[i] = float(rand())/float(RAND_MAX);
    }
    Tensor ltest2 = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)ltest2.data)[i] = float(rand())/float(RAND_MAX);
    }
    Tensor lweight = {{128}};
    for (int i = 0; i < 128; i++){
        ((float*)lweight.data)[i] = float(rand())/float(RAND_MAX);
    }

    // bfloat16 test
    Tensor ltest_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};

    for (int i = 0; i < 256; i++){
        ((bfloat16*)ltest_bf16.data)[i] = bfloat16(((float*)ltest.data)[i]);
    }
    Tensor ltest2_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256; i++){
        ((bfloat16*)ltest2_bf16.data)[i] = bfloat16(((float*)ltest2.data)[i]);
    }
    Tensor lweight_bf16 = {{128}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 128; i++){
        ((bfloat16*)lweight_bf16.data)[i] = bfloat16(((float*)lweight.data)[i]);
    }

    // cuda float test
    Tensor ltest_cuda = ltest.cuda();
    Tensor ltest2_cuda = ltest2.cuda();
    Tensor lweight_cuda = lweight.cuda();

    // cuda bfloat16 test
    Tensor ltest_bf16_cuda = ltest_bf16.cuda();
    Tensor ltest2_bf16_cuda = ltest2_bf16.cuda();
    Tensor lweight_bf16_cuda = lweight_bf16.cuda();

    auto ltest_out = lweight.lerp(ltest,ltest2);
    auto ltest_out_bf16 = lweight_bf16.lerp(ltest_bf16,ltest2_bf16);

    auto ltest_out_cuda = lweight_cuda.lerp(ltest_cuda,ltest2_cuda);
    auto ltest_out_bf16_cuda = lweight_bf16_cuda.lerp(ltest_bf16_cuda,ltest2_bf16_cuda);

    std::cout << "Lerp tests" << std::endl;
    std::cout << ltest_out << std::endl;
    std::cout << ltest_out_bf16 << std::endl;
    std::cout << ltest_out_cuda << std::endl;
    std::cout << ltest_out_bf16_cuda << std::endl;

    std::cout << ltest_out.get<float>(0) << "," << ltest_out.get<float>(1) << "," << ltest_out.get<float>(2) << "," << ltest_out.get<float>(3) << std::endl;
    std::cout << (float)ltest_out_bf16.get<bfloat16>(0) << "," << (float)ltest_out_bf16.get<bfloat16>(1) << "," << (float)ltest_out_bf16.get<bfloat16>(2) << "," << (float)ltest_out_bf16.get<bfloat16>(3) << std::endl;
    std::cout << ltest_out_cuda.get<float>(0) << "," << ltest_out_cuda.get<float>(1) << "," << ltest_out_cuda.get<float>(2) << "," << ltest_out_cuda.get<float>(3) << std::endl;
    std::cout << (float)ltest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)ltest_out_bf16_cuda.get<bfloat16>(3) << std::endl;
    
    // swishmul test
    Tensor wtest = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)wtest.data)[i] = float(rand())/float(RAND_MAX);
    }
    Tensor wtest2 = {{256}};
    for (int i = 0; i < 256; i++){
        ((float*)wtest2.data)[i] = float(rand())/float(RAND_MAX);
    }

    // bfloat16 test
    Tensor wtest_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};
    Tensor wtest2_bf16 = {{256}, TENSORTYPE::kBFLOAT_16};
    for (int i = 0; i < 256; i++){
        ((bfloat16*)wtest_bf16.data)[i] = bfloat16(((float*)wtest.data)[i]);
        ((bfloat16*)wtest2_bf16.data)[i] = bfloat16(((float*)wtest2.data)[i]);
    }
    // cuda float test
    Tensor wtest_cuda = wtest.cuda();
    Tensor wtest2_cuda = wtest2.cuda();

    // cuda bfloat16 test
    Tensor wtest_bf16_cuda = wtest_bf16.cuda();
    Tensor wtest2_bf16_cuda = wtest2_bf16.cuda();

    auto wtest_out = wtest.swishmul(wtest2);
    auto wtest_out_bf16 = wtest_bf16.swishmul(wtest2_bf16);

    auto wtest_out_cuda = wtest_cuda.swishmul(wtest2_cuda);
    auto wtest_out_bf16_cuda = wtest_bf16_cuda.swishmul(wtest2_bf16_cuda);

    std::cout << "Swishmul tests" << std::endl;
    std::cout << wtest_out << std::endl;
    std::cout << wtest_out_bf16 << std::endl;
    std::cout << wtest_out_cuda << std::endl;
    std::cout << wtest_out_bf16_cuda << std::endl;

    std::cout << wtest_out.get<float>(0) << "," << wtest_out.get<float>(1) << "," << wtest_out.get<float>(2) << "," << wtest_out.get<float>(3) << std::endl;
    std::cout << (float)wtest_out_bf16.get<bfloat16>(0) << "," << (float)wtest_out_bf16.get<bfloat16>(1) << "," << (float)wtest_out_bf16.get<bfloat16>(2) << "," << (float)wtest_out_bf16.get<bfloat16>(3) << std::endl;
    std::cout << wtest_out_cuda.get<float>(0) << "," << wtest_out_cuda.get<float>(1) << "," << wtest_out_cuda.get<float>(2) << "," << wtest_out_cuda.get<float>(3) << std::endl;
    std::cout << (float)wtest_out_bf16_cuda.get<bfloat16>(0) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(1) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(2) << "," << (float)wtest_out_bf16_cuda.get<bfloat16>(3) << std::endl;

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

    Tensor wkvtest_cuda = wkvtest.cuda();
    

    auto out = wkvtest.wkv5(k,v,r,decay,first);

    std::cout << "wkvtest" << std::endl;
    std::cout << out << std::endl;
    std::cout << out.get<float>(0) << ","<< out.get<float>(1) <<","<< out.get<float>(2) << ","<<out.get<float>(3) << std::endl;
    std::cout << wkvtest.get<float>(0) << ","<<wkvtest.get<float>(1) <<","<< wkvtest.get<float>(2) <<","<< wkvtest.get<float>(3) << std::endl;

    // wkvtest cuda
    Tensor k_cuda = k.cuda();
    Tensor v_cuda = v.cuda();
    Tensor r_cuda = r.cuda();
    Tensor decay_cuda = decay.cuda();
    Tensor first_cuda = first.cuda();

    auto out_cuda = wkvtest_cuda.wkv5(k_cuda,v_cuda,r_cuda,decay_cuda,first_cuda);

    std::cout << "wkvtest cuda" << std::endl;
    std::cout << out_cuda << std::endl;
    std::cout << out_cuda.get<float>(0) << ","<< out_cuda.get<float>(1) <<","<< out_cuda.get<float>(2) << ","<<out_cuda.get<float>(3) << std::endl;
    std::cout << wkvtest_cuda.get<float>(0) << ","<<wkvtest_cuda.get<float>(1) <<","<< wkvtest_cuda.get<float>(2) <<","<< wkvtest_cuda.get<float>(3) << std::endl;

    auto wkvtest_bf16_cuda = wkvtest_bf16.cuda();
    // wkvtest bf16
    auto out_bf16 = wkvtest_bf16.wkv5(kbf16,vbf16,rbf16,decaybf16,firstbf16);

    std::cout << "wkvtest bf16" << std::endl;
    std::cout << out_bf16 << std::endl;
    std::cout << wkvtest_bf16 << std::endl;
    // wkvtest cuda bf16
    auto kbf16_cuda = kbf16.cuda();
    auto vbf16_cuda = vbf16.cuda();
    auto rbf16_cuda = rbf16.cuda();
    auto decaybf16_cuda = decaybf16.cuda();
    auto firstbf16_cuda = firstbf16.cuda();

    auto out_bf16_cuda = wkvtest_bf16_cuda.wkv5(kbf16_cuda,vbf16_cuda,rbf16_cuda,decaybf16_cuda,firstbf16_cuda);

    std::cout << "wkvtest cuda bf16" << std::endl;
    std::cout << out_bf16_cuda << std::endl;
    std::cout << wkvtest_bf16_cuda << std::endl;
    
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
    std::cout << "matmultest cuda" << std::endl;
    Tensor matmultest_cuda = matmultest.cuda();
    Tensor matmultest2_cuda = matmultest2.cuda();

    std::cout << "Matmultestprematmul cuda" << std::endl;

    auto outputmm_cuda = matmultest_cuda.matmul(matmultest2_cuda);
    
    std::cout << "matmultest bf16 cuda" << std::endl;

    Tensor matmultest_bf16_cuda = matmultest_bf16.cuda();
    Tensor matmultest2_bf16_cuda = matmultest2_bf16.cuda();

    std::cout << "Matmultestprematmul bf16 cuda" << std::endl;

    auto outputmm_bf16_cuda = matmultest_bf16_cuda.matmul(matmultest2_bf16_cuda);

    std::cout << "matmultest2 cuda" << std::endl;
    std::cout << outputmm_cuda << std::endl;
    std::cout << outputmm_bf16_cuda << std::endl;
    std::cout << outputmm_cuda.get<float>(0) << ","<< outputmm_cuda.get<float>(1) <<","<< outputmm_cuda.get<float>(2) << ","<<outputmm_cuda.get<float>(3) << std::endl;
    std::cout << outputmm_bf16_cuda.get<bfloat16>(0) << ","<< outputmm_bf16_cuda.get<bfloat16>(1) <<","<< outputmm_bf16_cuda.get<bfloat16>(2) << ","<<outputmm_bf16_cuda.get<bfloat16>(3) << std::endl;

    // matmultest cuda

    Tensor a = {{4096,2048},TENSORTYPE::kBFLOAT_16};
    for(size_t i = 0; i < 2048*4096; i++){
        ((bfloat16*)a.data)[i] = bfloat16(0.25*float(rand())/float(RAND_MAX));
    }

    Tensor b = {{1,18,2048},TENSORTYPE::kBFLOAT_16};
    for(size_t i = 0; i < 18*2048; i++){
        ((bfloat16*)b.data)[i] =  bfloat16(0.25*float(rand())/float(RAND_MAX));;
    }

    auto outcpu = a.matmul(b);

    std::cout << outcpu;

    auto acuda = a.cuda();
    auto bcuda = b.cuda();

    auto outcuda = acuda.matmul(bcuda);

    std::cout << outcuda;

    return 0;
}