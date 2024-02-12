
#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H
#include <iostream>
#include "nlohmann/json.hpp"
#if defined(__ARM_NEON)
#define ARMONLY(x) x
#define AVXONLY(x)
#include <arm_neon.h>
#define NEONBF16 __attribute__ ((target ("bf16")))
#else
#define ARMONLY(x) 
#define AVXONLY(x) x
#define NEONBF16
#endif


/*
Define 
#pragma CUDAONLY
function() {
    // code
}
to remove the function from the CPU build
*/
#if defined(__CUDACC__)
#define CUDAONLY
#else
#define CUDAONLY if (false)
#define cudaMemcpy(...) throw std::runtime_error("Not compiled with cuda")
#define cudaMalloc(...) throw std::runtime_error("Not compiled with cuda")
#define cudaMemset(...) throw std::runtime_error("Not compiled with cuda")
#define cudaMallocHost(pointer, size) *pointer = malloc(size)


#endif

// if windows, define posix_memalign
#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
static int posix_memalign(void **memptr, size_t alignment, size_t size){
    *memptr = _aligned_malloc(size, alignment);
    return 0;
}
#endif

// backtrace
#if defined(__CUDACC__) || defined(__HIPCC__)
#include "cuda_runtime.h"

static void check_for_errors (){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        std::cout << "CUDA error: " << cudaGetErrorName(error) << " " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
    }
}

#else
static void check_for_errors (){
    // do nothing
}
#endif


// define float16 and bfloat16
// typedef unsigned short float16;
// typedef unsigned short bfloat16;

struct float16{
    uint16_t fvalue;
    operator uint16_t() const {return fvalue;}
};


#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

#define bfloat16 __bf16


static float bfloat16_to_float32(bfloat16 value){
    auto x = (uint32_t(*(uint16_t*)(&value))<<16);
    return *((float*)&x);
}

#elif defined(__CUDACC__)


#include <cuda_bf16.h>
#define bfloat16 __nv_bfloat16
static float bfloat16_to_float32(bfloat16 value){

    return float(value);

}

#else

#define BF16FALLBACKS

struct bfloat16;
static float bfloat16_to_float32(bfloat16 value);
static bfloat16 float32_to_bfloat16(float value);
struct bfloat16{
    uint16_t value;
    operator float() const {return bfloat16_to_float32(*this);}
    bfloat16(double valuein) {this->value = float32_to_bfloat16((float)valuein);}
    bfloat16(float valuein) {this->value = float32_to_bfloat16(valuein);}
    bfloat16(uint16_t valuein) {this->value = valuein;}
    bfloat16() {this->value = 0;}
    bfloat16 operator = (float valuein) {this->value = float32_to_bfloat16(valuein); return *this;}
    bfloat16 operator = (bfloat16 valuein) {this->value = valuein.value; return *this;}
    // bfloat16 operator = (uint16_t value) {this->value = value; return *this;}
    // bfloat16 operator = (double value) {this->value = float32_to_bfloat16((float)value); return *this;}
    bfloat16 operator + (bfloat16 valuein) {return bfloat16(float(*this) + float(valuein));}
    
};



static float bfloat16_to_float32(bfloat16 value){
    // cast as uint16_t, then cast as float32, then bitshift 16 bits to the left, then cast as float32
    uint32_t inter (uint32_t((uint16_t)value.value) << 16);
    return *((float*)&inter);
}

static bfloat16 float32_to_bfloat16(float value){
    // cast as uint32_t, then bitshift 16 bits to the right, then cast as uint16_t, then cast as bfloat16
    uint32_t inter (uint32_t(*((uint32_t*)&value)) >> 16);
    return {
        (uint16_t)inter
    };
}

#endif






#define flp(x) ((float*)(x))
#define bflp(x) ((bfloat16*)(x))


static std::ostream& operator<<(std::ostream& os, const bfloat16& value){
    return os << bfloat16_to_float32(value);
}
// bf16 chevrons for std::cout


enum TENSORTYPE
{
    /// Boolean type
    kBOOL,
    /// Unsigned byte
    kUINT_8,
    /// Signed byte
    kINT_8,
    /// Signed integer (16-bit)
    kINT_16,
    /// Unsigned integer (16-bit)
    kUINT_16,
    /// Half-precision floating point
    kFLOAT_16,
    /// Brain floating point
    kBFLOAT_16,
    /// Signed integer (32-bit)
    kINT_32,
    /// Unsigned integer (32-bit)
    kUINT_32,
    /// Floating point (32-bit)
    kFLOAT_32,
    /// Floating point (64-bit)
    kFLOAT_64,
    /// Signed integer (64-bit)
    kINT_64,
    /// Unsigned integer (64-bit)
    kUINT_64,

};

NLOHMANN_JSON_SERIALIZE_ENUM(TENSORTYPE, {
                                             {kBOOL, "BOOL"},
                                             {kUINT_8, "U8"},
                                             {kINT_8, "I8"},
                                             {kINT_16, "I16"},
                                             {kUINT_16, "U16"},
                                             {kFLOAT_16, "F16"},
                                             {kBFLOAT_16, "BF16"},
                                             {kINT_32, "I32"},
                                             {kUINT_32, "U32"},
                                             {kFLOAT_32, "F32"},
                                             {kFLOAT_64, "F64"},
                                             {kINT_64, "I64"},
                                             {kUINT_64, "U64"},
                                         })

enum DEVICE {
    CPU,
    CUDA,
    ROCM,
    VULKAN
};

NLOHMANN_JSON_SERIALIZE_ENUM(DEVICE, {
                                             {CPU, "CPU"},
                                             {CUDA, "CUDA"},
                                             {ROCM, "ROCM"},
                                             {VULKAN, "VULKAN"},
                                         })

static std::string get_device_name(DEVICE device) {
    switch (device){
        case CPU:
            return "CPU";
        case CUDA:
            return "CUDA";
        case ROCM:
            return "ROCM";
        case VULKAN:
            return "VULKAN";
        default:
            return "UNKNOWN";
    }
}

static std::string get_dtype_name(TENSORTYPE dtype){
    switch (dtype){
        case kBOOL:
            return "BOOL";
        case kUINT_8:
            return "U8";
        case kINT_8:
            return "I8";
        case kINT_16:
            return "I16";
        case kUINT_16:
            return "U16";
        case kFLOAT_16:
            return "F16";
        case kBFLOAT_16:
            return "BF16";
        case kINT_32:
            return "I32";
        case kUINT_32:
            return "U32";
        case kFLOAT_32:
            return "F32";
        case kFLOAT_64:
            return "F64";
        case kINT_64:
            return "I64";
        case kUINT_64:
            return "U64";
        default:
            return "UNKNOWN";
    }
}

static size_t get_dtype_bytes(TENSORTYPE dtype){
    switch (dtype){
        case kBOOL:
            return 1;
        case kUINT_8:
            return 1;
        case kINT_8:
            return 1;
        case kINT_16:
            return 2;
        case kUINT_16:
            return 2;
        case kFLOAT_16:
            return 2;
        case kBFLOAT_16:
            return 2;
        case kINT_32:
            return 4;
        case kUINT_32:
            return 4;
        case kFLOAT_32:
            return 4;
        case kFLOAT_64:
            return 8;
        case kINT_64:
            return 8;
        case kUINT_64:
            return 8;
        default:
            return 0;
    }
};

template <typename T>
static TENSORTYPE get_tensortype(){
    if (std::is_same<T, bool>::value){
        return kBOOL;
    } else if (std::is_same<T, uint8_t>::value){
        return kUINT_8;
    } else if (std::is_same<T, int8_t>::value){
        return kINT_8;
    } else if (std::is_same<T, int16_t>::value){
        return kINT_16;
    } else if (std::is_same<T, uint16_t>::value){
        return kUINT_16;
    } else if (std::is_same<T, bfloat16>::value){
        return kBFLOAT_16;
    } else if (std::is_same<T, float16>::value){
        return kFLOAT_16;
    } else if (std::is_same<T, int32_t>::value){
        return kINT_32;
    } else if (std::is_same<T, uint32_t>::value){
        return kUINT_32;
    } else if (std::is_same<T, float>::value){
        return kFLOAT_32;
    } else if (std::is_same<T, double>::value){
        return kFLOAT_64;
    } else if (std::is_same<T, int64_t>::value){
        return kINT_64;
    } else if (std::is_same<T, uint64_t>::value){
        return kUINT_64;
    } else {
        return kBOOL;
    }
};


struct Tensor{
    std::vector<size_t> shape;
    size_t data_size_in_bytes;
    TENSORTYPE dtype;
    DEVICE device;
    int device_id;
    void* data = nullptr;
    Tensor(){
        shape = std::vector<size_t>();
        data_size_in_bytes = 0;
        dtype = kFLOAT_32;
        device = CPU;
        device_id = 0;
        data = nullptr;
    }
    Tensor(std::vector<size_t> shape, TENSORTYPE dtype=TENSORTYPE::kFLOAT_32, DEVICE device=DEVICE::CPU, int device_id=0){
        this->shape = shape;
        this->dtype = dtype;
        this->device = device;
        this->device_id = device_id;
        this->data_size_in_bytes = get_dtype_bytes(dtype);
        for (size_t i = 0; i < shape.size(); i++){
            this->data_size_in_bytes *= shape[i];
        }

        if (device == DEVICE::CUDA){
            cudaMalloc(&this->data, this->data_size_in_bytes);
            cudaMemset(this->data, 0, this->data_size_in_bytes);
        } else if (device == DEVICE::ROCM){
            cudaMalloc(&this->data, this->data_size_in_bytes);
        } else if (device == DEVICE::VULKAN){
            cudaMalloc(&this->data, this->data_size_in_bytes);
        } else {
            posix_memalign(&this->data,64,this->data_size_in_bytes);
            // cudaMallocHost(&this->data,this->data_size_in_bytes);

            
            assert(this->data != nullptr);

            // fill with zeros
            memset(this->data, 0, this->data_size_in_bytes);
        }

        // this->data = malloc(this->data_size_in_bytes);

        // print current stack trace

    }

    Tensor(std::vector<size_t> shape, void* data, TENSORTYPE dtype=TENSORTYPE::kFLOAT_32, DEVICE device=DEVICE::CPU, int device_id=0){
        this->shape = shape;
        this->dtype = dtype;
        this->device = device;
        this->device_id = device_id;
        this->data_size_in_bytes = get_dtype_bytes(dtype);
        for (size_t i = 0; i < shape.size(); i++){
            this->data_size_in_bytes *= shape[i];
        }
        this->data = data;
    }

    void empty(){
        if (data != nullptr){
            if (device == DEVICE::CUDA){
                cudaMemset(data, 0, data_size_in_bytes);
            } else if (device == DEVICE::CPU){
                memset(data, 0, data_size_in_bytes);
            } else{
                throw std::runtime_error("empty only implemented for CPU and CUDA");
            }
        }
    }

    Tensor cloneWithFalseReshape( std::vector<size_t> newshape){
        Tensor new_tensor = Tensor();
        new_tensor.shape = newshape;
        new_tensor.dtype = dtype;
        new_tensor.device = device;
        new_tensor.device_id = device_id;
        new_tensor.data_size_in_bytes = get_dtype_bytes(dtype);
        for (size_t i = 0; i < newshape.size(); i++){
            new_tensor.data_size_in_bytes *= newshape[i];
        }
        if (new_tensor.data_size_in_bytes > data_size_in_bytes){
            throw std::runtime_error("cloneWithFalseReshape must have same or smaller size, to avoid memory access issues");
        }
        new_tensor.data = data;
        return new_tensor;
    }

    template <typename T>
    Tensor(std::vector<size_t> shape, std::vector<T> data, DEVICE device=DEVICE::CPU, int device_id=0){
        this->shape = shape;
        this->dtype = get_tensortype<T>();
        this->device = device;
        this->device_id = device_id;
        this->data_size_in_bytes = get_dtype_bytes(dtype);
        for (size_t i = 0; i < shape.size(); i++){
            this->data_size_in_bytes *= shape[i];
        }
        posix_memalign(&this->data ,64,this->data_size_in_bytes);
        memcpy(this->data, data.data(), this->data_size_in_bytes);

    }

    Tensor(const Tensor& other){
        this->shape = other.shape;
        this->dtype = other.dtype;
        this->device = other.device;
        this->device_id = other.device_id;
        this->data_size_in_bytes = other.data_size_in_bytes;
        this->data = other.data;
    }

    // copy assignment
    Tensor& operator=(const Tensor& other){
        this->shape = other.shape;
        this->dtype = other.dtype;
        this->device = other.device;
        this->device_id = other.device_id;
        this->data_size_in_bytes = other.data_size_in_bytes;
        this->data = other.data;
        return *this;
    }
    

    template <typename T>
    T get(size_t index)const{
        if (device == DEVICE::CUDA){
            T value;
            cudaMemcpy(&value, (void*)((T*)data + index), get_dtype_bytes(dtype), cudaMemcpyDeviceToHost);
            return value;
        } else if (device == DEVICE::ROCM){
            T value;
            cudaMemcpy(&value, (void*)((T*)data + index), get_dtype_bytes(dtype), cudaMemcpyDeviceToHost);
            return value;
        } else if (device == DEVICE::VULKAN){
            T value;
            cudaMemcpy(&value, (void*)((T*)data + index), get_dtype_bytes(dtype), cudaMemcpyDeviceToHost);
            return value;
        } else {
            return ((T*)data)[index];
        }
    }

    Tensor cuda(){
        if (device == DEVICE::CUDA){
            return *this;
        } else {
            Tensor new_tensor = Tensor(shape, dtype, DEVICE::CUDA, device_id);
            if(data !=nullptr){
                cudaMemcpy(new_tensor.data, data, data_size_in_bytes, cudaMemcpyHostToDevice);
            }
            return new_tensor;
        }
    }

    Tensor cpu(){
        if (device == DEVICE::CPU){
            return *this;
        } else {
            Tensor new_tensor = Tensor(shape, dtype, DEVICE::CPU, device_id);
            cudaMemcpy(new_tensor.data, data, data_size_in_bytes, cudaMemcpyDeviceToHost);
            return new_tensor;
        }
    }

    Tensor float32(){
        if (dtype == TENSORTYPE::kFLOAT_32){
            return *this;
        } else if (dtype == TENSORTYPE::kBFLOAT_16){
            Tensor new_tensor = Tensor(shape, TENSORTYPE::kFLOAT_32, device, device_id);

            for (size_t i = 0; i < get_element_count(); i++){
                ((float*)new_tensor.data)[i] = bfloat16_to_float32(((bfloat16*)data)[i]);
            }
            
            return new_tensor;
        }
        else{
            throw std::runtime_error("float32 only implemented for float and bfloat16");
        }
    }

    size_t get_element_count() const {
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); i++){
            count *= shape[i];
        }
        return count;
    }

    Tensor relusquared();
    Tensor sigmoidmul(Tensor& other, Tensor& residual);
    Tensor lerp(Tensor& A, Tensor& B, Tensor& C);
    Tensor swishmul(Tensor& other);
    Tensor matmul(Tensor& other, Tensor residual = Tensor());
    Tensor matmul(Tensor &Art, Tensor &Aot, Tensor &Bt, Tensor residual = Tensor());
    Tensor normalize(Tensor& weight, Tensor& bias, Tensor& result, size_t heads = 1, float epsilon=1e-5);
    
    Tensor wkv5(Tensor& r, Tensor& k, Tensor& v, Tensor& w, Tensor& u, Tensor& y);

    Tensor operator[](size_t index);
    Tensor gather(std::vector<std::vector<size_t>> index, Tensor result);
    
    void copyfrom(Tensor other){
        if (device == DEVICE::CUDA){
            cudaMemcpy(data, other.data, data_size_in_bytes, cudaMemcpyDeviceToDevice);
        } else if (device == DEVICE::ROCM){
            cudaMemcpy(data, other.data, data_size_in_bytes, cudaMemcpyDeviceToDevice);
        } else if (device == DEVICE::VULKAN){
            cudaMemcpy(data, other.data, data_size_in_bytes, cudaMemcpyDeviceToDevice);
        } else {
            memcpy(data, other.data, data_size_in_bytes);
        }
    }
    
    template <typename T>
    Tensor operator=(T value){
        // if T is a tensor, copy data
        if (std::is_same<T, Tensor>::value){
            this->data = ((Tensor*)&value)->data;
            this->data_size_in_bytes = ((Tensor*)&value)->data_size_in_bytes;
            this->dtype = ((Tensor*)&value)->dtype;
            this->shape = ((Tensor*)&value)->shape;
            this->device = ((Tensor*)&value)->device;
            this->device_id = ((Tensor*)&value)->device_id;
            return *this;
        }
        
        // make sure size is {1} or {}
        if ((shape.size() == 1 && shape[0] == 1) || shape.size() == 0){
            if (dtype == kFLOAT_32)
                ((float*)data)[0] = value;
            else if (dtype == kFLOAT_64)
                ((double*)data)[0] = value;
            else if (dtype == kINT_32)
                ((int32_t*)data)[0] = value;
            else if (dtype == kINT_64)
                ((int64_t*)data)[0] = value;
            else if (dtype == kUINT_32)
                ((uint32_t*)data)[0] = value;
            else if (dtype == kUINT_64)
                ((uint64_t*)data)[0] = value;
            else if (dtype == kINT_8)
                ((int8_t*)data)[0] = value;
            else if (dtype == kUINT_8)
                ((uint8_t*)data)[0] = value;
            else if (dtype == kINT_16)
                ((int16_t*)data)[0] = value;
            else if (dtype == kUINT_16)
                ((uint16_t*)data)[0] = value;
            else if (dtype == kFLOAT_16)
                ((float16*)data)[0].fvalue = value;
            else if (dtype == kBFLOAT_16)
                ((bfloat16*)data)[0] = value;
            else if (dtype == kBOOL)
                ((bool*)data)[0] = value;
            else {
                throw std::runtime_error("Unknown dtype");
            }
        } else {
            throw std::runtime_error("Tensor must be of size {1} or {}");
        }
        return *this;
    }

    inline Tensor reshape(std::vector<size_t> shape);
};

#include "tensor/operators/ops.h"

#endif //TENSOR_TENSOR_H