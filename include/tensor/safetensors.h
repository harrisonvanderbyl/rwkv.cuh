//
// Created by mfuntowicz on 3/28/23.
//

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <span>
#include <fstream>
#include "tensor/tensor.h"

using json = nlohmann::json;


    
    struct metadata_t {
        TENSORTYPE dtype;
        std::vector<size_t> shape;
        std::pair<size_t, size_t> data_offsets;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata_t, dtype, shape, data_offsets)

    /**
     *
     */
    class safetensors {

    public:

        std::unordered_map<std::string, const metadata_t> metas;
        
       
        const char* storage = nullptr;
        /**
         *
         * @return
         */
        inline size_t size() const { return metas.size(); }

        /**
         *
         * @param name
         * @return
         */
        
         Tensor operator[](const char *name) const {
                const auto& meta = metas.at(name);
                char* data_begin = const_cast<char*>(storage) + meta.data_offsets.first;
                // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;
                
                Tensor tensor = {meta.shape, data_begin, meta.dtype, DEVICE::CPU, 0};
                return tensor;
            }
         Tensor operator[](std::string name) const{
                return operator[](name.c_str());
         }

         /**
         *
         * @param name
         * @return
         */
        inline std::vector<const char*> keys() const {
            std::vector<const char*> keys;
            keys.reserve(metas.size());
            for (auto &item: metas) {
                keys.push_back(item.first.c_str());
            }
            return keys;
        }

        // contains key
        inline bool contains(const char* name) const {
            auto keys = this->keys();
            bool found = false;

            for (auto key : keys){
                if (strcmp(key, name) == 0){
                    found = true;
                }

            }
            return found;
        }
        inline bool contains(std::string name) const {
            return contains(name.c_str());
        }

        safetensors(){};

        safetensors(std::basic_istream<char> &in) {
                uint64_t header_size = 0;

                // todo: handle exception
                in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

                std::vector<char> meta_block(header_size);
                in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
                const auto metadatas = json::parse(meta_block);

                // How many bytes remaining to pre-allocate the storage tensor
                in.seekg(0, std::ios::end);
                std::streamsize f_size = in.tellg();
                in.seekg(8 + header_size, std::ios::beg);
                const auto tensors_size = f_size - 8 - header_size;

                metas = std::unordered_map<std::string, const metadata_t>(metadatas.size());
                // allocate in a way that prevents it from being freed
                // storage = new char[tensors_size];
                posix_memalign((void**)&storage, 128, tensors_size);
                

                // Read the remaining content
                in.read((char*)storage, static_cast<std::streamsize>(tensors_size));

                // Populate the meta lookup table
                if (metadatas.is_object()) {
                    for (auto &item: metadatas.items()) {
                        if (item.key() != "__metadata__") {
                            const auto name = std::string(item.key());
                            const auto& info = item.value();

                            const metadata_t meta = {info["dtype"].get<TENSORTYPE>(), info["shape"], info["data_offsets"]};
                            metas.insert(std::pair<std::string, metadata_t>(name, meta));
                        }
                    }
                }

            }


            safetensors(const char* filename) {
                std::ifstream bin(filename, std::ios::binary);
                *this = safetensors(bin);
            }

            };





    

    


#endif //SAFETENSORS_H