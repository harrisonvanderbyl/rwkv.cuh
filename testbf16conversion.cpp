#include "cpuops.cpp"
#include <iostream>

__attribute__ ((target ("avx512bf16")))
int main(){
    Tensor input({32}, TENSORTYPE::kFLOAT_32);
    Tensor other({32}, TENSORTYPE::kFLOAT_32);
    Tensor input16({32}, TENSORTYPE::kBFLOAT_16);
    Tensor other16({32}, TENSORTYPE::kBFLOAT_16);

    std::cout << "bf16_float32 fill start" << std::endl;

    for (size_t i = 0; i < 32; i++){
        ((float*)(input.data))[i] = float(rand()) / float(RAND_MAX);
        ((float*)(other.data))[i] = float(rand()) / float(RAND_MAX);
        ((bfloat16*)(input16.data))[i] = ((float*)(input.data))[i];
        ((bfloat16*)(other16.data))[i] = ((float*)(other.data))[i];
    }


    std::cout << "bf16_float32 conversion start" << std::endl;

    auto m = bf16_float32(_mm256_loadu_si256((__m256i*)input16.data));
    auto m2 = bf16_float32(_mm256_loadu_si256((__m256i*)(bflp(input16.data) + 8)));

    std::cout << "bf16_float32 conversion" << std::endl;

    std::cout << m[0] << "," << input.get<float>(0) << std::endl;
    std::cout << m[1] << "," << input.get<float>(1) << std::endl;
    std::cout << m2[0] << "," << input.get<float>(8) << std::endl;
    std::cout << m2[1] << "," << input.get<float>(9) << std::endl;

    std::cout << "bf16_float32 conversion end" << std::endl;

    std::cout << "float32_bf16 conversion start" << std::endl;

    auto m3 = float32_bf16(_mm512_loadu_ps(flp(input.data)+16), _mm512_loadu_ps(flp(input.data)+0));
    bfloat16* m4 = (bfloat16*)(&m3);

    std::cout << "float32_bf16 conversion" << std::endl;
    
    std::cout << float(((m4))[0]) << "," << float(input16.get<bfloat16>(0)) << std::endl;
    std::cout << float(((m4))[1]) << "," << float(input16.get<bfloat16>(1)) << std::endl;
    std::cout << float(((m4))[16]) << "," << float(input16.get<bfloat16>(16)) << std::endl;
    std::cout << float(((m4))[17]) << "," << float(input16.get<bfloat16>(17)) << std::endl;
    
    // avx2 test
    std::cout << "avx2 test" << std::endl;

    auto m5 = bf16_float32_avx2(_mm_loadu_si128((__m128i*)input16.data));
    auto m51 = bf16_float32_avx2(_mm_loadu_si128((__m128i*)(bflp(input16.data) + 8)));
    auto m6 = float32_bf16_avx2(_mm256_loadu_ps(flp(input.data)+8), _mm256_loadu_ps(flp(input.data)+0));

    int test[8] = {1,2,3,4,5,6,7,8};
    int test2[8] = {9,10,11,12,13,14,15,16};
    auto test3 = _mm256_packs_epi32(_mm256_loadu_si256((__m256i*)test), _mm256_loadu_si256((__m256i*)test2)); 
    test3 = _mm256_permute4x64_epi64(test3, 0b11011000);
    std::cout <<
    ((short*)&test3)[0] << "," <<
    ((short*)&test3)[1] << "," <<
    ((short*)&test3)[2] << "," <<
    ((short*)&test3)[3] << "," <<
    ((short*)&test3)[4] << "," <<
    ((short*)&test3)[5] << "," <<
    ((short*)&test3)[6] << "," <<
    ((short*)&test3)[7] << "," <<
    ((short*)&test3)[8] << "," <<
    ((short*)&test3)[9] << "," <<
    ((short*)&test3)[10] << "," <<
    ((short*)&test3)[11] << "," <<
    ((short*)&test3)[12] << "," <<
    ((short*)&test3)[13] << "," <<
    ((short*)&test3)[14] << "," <<
    ((short*)&test3)[15] << "," <<
     std::endl;


    std::cout << "avx2 test" << std::endl;
    std::cout << input.get<float>(0) << "," << m5[0] << "," << float(((bfloat16*)&m6)[0]) << std::endl;
    std::cout << input.get<float>(1) << "," << m5[1] << "," << float(((bfloat16*)&m6)[1]) << std::endl;
    std::cout << input.get<float>(2) << "," << m5[2] << "," << float(((bfloat16*)&m6)[2]) << std::endl;
    std::cout << input.get<float>(3) << "," << m5[3] << "," << float(((bfloat16*)&m6)[3]) << std::endl;
    std::cout << input.get<float>(4) << "," << m5[4] << "," << float(((bfloat16*)&m6)[4]) << std::endl;
    std::cout << input.get<float>(5) << "," << m5[5] << "," << float(((bfloat16*)&m6)[5]) << std::endl;
    std::cout << input.get<float>(6) << "," << m5[6] << "," << float(((bfloat16*)&m6)[6]) << std::endl;
    std::cout << input.get<float>(7) << "," << m5[7] << "," << float(((bfloat16*)&m6)[7]) << std::endl;
    std::cout << input.get<float>(8) << "," << m51[0] << "," << float(((bfloat16*)&m6)[8]) << std::endl;
    std::cout << input.get<float>(9) << "," << m51[1] << "," << float(((bfloat16*)&m6)[9]) << std::endl;
    std::cout << input.get<float>(10) << "," << m51[2] << "," << float(((bfloat16*)&m6)[10]) << std::endl;
    std::cout << input.get<float>(11) << "," << m51[3] << "," << float(((bfloat16*)&m6)[11]) << std::endl;
    std::cout << input.get<float>(12) << "," << m51[4] << "," << float(((bfloat16*)&m6)[12]) << std::endl;
    std::cout << input.get<float>(13) << "," << m51[5] << "," << float(((bfloat16*)&m6)[13]) << std::endl;
    std::cout << input.get<float>(14) << "," << m51[6] << "," << float(((bfloat16*)&m6)[14]) << std::endl;
    std::cout << input.get<float>(15) << "," << m51[7] << "," << float(((bfloat16*)&m6)[15]) << std::endl;

}