#ifndef BYTEMATCH_HPP
#define BYTEMATCH_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include "tokenizer/bytematch.hpp"
#include <string.h>


void chr_cstrlit(unsigned char u, char *buffer, size_t buflen)
{
    if (buflen < 2)
        *buffer = '\0';
    else if (isprint(u) && u != '\'' && u != '\"' && u != '\\' && u != '\?')
        sprintf(buffer, "%c", u);
    else if (buflen < 3)
        *buffer = '\0';
    else
    {
        switch (u)
        {
        case '\a':  strcpy(buffer, "\\a"); break;
        case '\b':  strcpy(buffer, "\\b"); break;
        case '\f':  strcpy(buffer, "\\f"); break;
        case '\n':  strcpy(buffer, "\\n"); break;
        case '\r':  strcpy(buffer, "\\r"); break;
        case '\t':  strcpy(buffer, "\\t"); break;
        case '\v':  strcpy(buffer, "\\v"); break;
        case '\\':  strcpy(buffer, "\\\\"); break;
        case '\'':  strcpy(buffer, "\\'"); break;
        case '\"':  strcpy(buffer, "\\\""); break;
        case '\?':  strcpy(buffer, "\\\?"); break;
        default:
            if (buflen < 5)
                *buffer = '\0';
            else
                sprintf(buffer, "\\%03o", u);
            break;
        }
    }
}

void str_cstrlit(const char *str, char *buffer, size_t buflen)
{
    unsigned char u;
    size_t len;

    while ((u = (unsigned char)*str++) != '\0')
    {
        chr_cstrlit(u, buffer, buflen);
        if ((len = strlen(buffer)) == 0)
            return;
        buffer += len;
        buflen -= len;
    }
    *buffer = '\0';
}


class RWKVTokenizer {
private:
    std::vector<std::vector<std::vector<std::vector<uchar>>>> table;
    std::vector<std::set<int>> good;
    std::vector<int> wlen;
    std::unordered_map<int, std::vector<uchar>> idx2token;
    std::unordered_map<std::string, int> token2idx;
    

public:
    RWKVTokenizer(const std::string& fileName) {
        // Reading file and constructing idx2token and token2idx
        std::ifstream file(fileName);
        std::string line;
        std::vector<std::vector<uchar>> sorted;
        std::vector<std::string> filelines;
        while (std::getline(file, line)) {
            // get progress without changing the file position
            // float progress = (float)file.tellg() / (float)std::filesystem::file_size(fileName);
            // if ((size_t)(progress * 100) > lastintprog) {
            //     lastintprog = (size_t)(progress * 100);
            //     // flush
            //     std::cout.flush();
            //     std::cout << "%\r" << "Loading token file:[" << std::string(lastintprog / 2, '=') << std::string(50 - lastintprog / 2, ' ') << "] " << lastintprog ;
                
            // }

            filelines.push_back(std::move(line));
        }

        std::cout << std::endl;
        
        
        // #pragma omp parallel for schedule(static, 1) shared(filelines, sorted, idx2token)
        for (size_t io = 0; io < filelines.size(); io++) {
            
            //progress
            

            std::string nline = filelines[io];
            size_t idxSpace = nline.find(' ');
            int idx = std::stoi(nline.substr(0, idxSpace));
            auto x = findPythonByteObjects(nline.substr(idxSpace + 1, nline.rfind(' ') - idxSpace - 1));
            if (x.size() == 0) {
                // decode the string as utf-8
                auto nline2 = nline.substr(idxSpace + 2, nline.rfind(' ') - idxSpace - 3);
                // std::cout << nline2 << std::endl;
                
                x = std::vector<uchar>(nline2.begin(), nline2.end());
            }

            // for (size_t i = 0; i < x.size(); i++) {
            // unescape the string
            // char buffer[5];
            // str_cstrlit((char*)x.data(), buffer, 5);
            // }

            // if x = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0" make it into the string representation
            if (std::string(x.begin(), x.end()) == "\\xa0\\xa0\\xa0\\xa0\\xa0\\xa0\\xa0\\xa0") {
                std::string xtr = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x = "\xa0\xa0\xa0\xa0" make it into the string representation
            if (std::string(x.begin(), x.end()) == "\\xa0\\xa0\\xa0\\xa0") {
                std::string xtr = "\xa0\xa0\xa0\xa0";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x = "\xa0\xa0\xa0" make it into the string representation
            if (std::string(x.begin(), x.end()) == "\\xa0\\xa0\\xa0") {
                std::string xtr = "\xa0\xa0\xa0";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x = "\xa0\xa0" make it into the string representation
            if (std::string(x.begin(), x.end()) == "\\xa0\\xa0") {
                std::string xtr = "\xa0\xa0";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x = " \xa0" make it into the string representation
            if (std::string(x.begin(), x.end()) == " \\xa0") {
                std::string xtr = " \xa0";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x = " \xad " make it into the string representation
            if (std::string(x.begin(), x.end()) == " \\xad ") {
                std::string xtr = " \xad ";
                x = std::vector<uchar>(xtr.begin(), xtr.end());
            }

            // if x starts with \x then it is a byte string
            if (x[0] == '\\' && x[1] == 'x'){
                // convert to hex
                std::string hexstr(x.begin() + 2, x.end());
                std::stringstream ss;
                ss << std::hex << hexstr;
                unsigned int n;
                ss >> n;
                x = std::vector<uchar>(1, n);
            }

            std::string str = std::string(x.begin(), x.end());
            // replace all \n with \n
            while (str.find("\\n") != std::string::npos)
            str.replace(str.find("\\n"), 2, "\n");
            // replace all \t with \t
            while (str.find("\\t") != std::string::npos)
            str.replace(str.find("\\t"), 2, "\t");
            // replace all \r with \r
            while (str.find("\\r") != std::string::npos)
            str.replace(str.find("\\r"), 2, "\r");

            // replace all \x with \x

            // replace all 
            while (str.find("\\\\") != std::string::npos)
            str.replace(str.find("\\\\"), 2, "[slashslash]");

            // replace all [slashslash] with slash
            while (str.find("[slashslash]") != std::string::npos)
            str.replace(str.find("[slashslash]"), 12, "\\");

            // replace all \uXXXX with the unicode character
            while (str.find("\\u") != std::string::npos) {
                std::string hexstr(str.begin() + str.find("\\u") + 2, str.begin() + str.find("\\u") + 6);
                std::stringstream ss;
                ss << std::hex << hexstr;
                unsigned int n;
                ss >> n;
                char buffer[5];
                chr_cstrlit(n, buffer, 5);
                str.replace(str.find("\\u"), 6, buffer);
            }


            x = std::vector<uchar>(str.begin(), str.end());

            
            // sorted.push_back(x)
            // #pragma omp critical
            {
                sorted.push_back(x);
            }
            
        
            // idx2token[idx] = x;
            // #pragma omp critical
            {
                idx2token[idx] = x;
            }
        }
        std::cout << std::endl;
        file.close();

        // Constructing token2idx
        for (const auto& pair : idx2token) {
            std::string tokenStr(pair.second.begin(), pair.second.end());
            token2idx[tokenStr] = pair.first;
        }

        // precompute some tables for fast matching
        // this.table = Array.from({ length: 256 }, () => Array.from({ length: 256 }, () => []));
        // this.good = Array.from({ length: 256 }, () => new Set<number>());
        // this.wlen = Array.from({ length: 256 }, () => 0);

        // for (let i = sorted.length - 1; i >= 0; i--) { // reverse order - match longer tokens first
        //     const s = sorted[i];
        //     if (s.length >= 2) {
        //         const s0 = s[0];
        //         const s1 = s[1];
        //         this.table[s0][s1].push(s);
        //         this.wlen[s0] = Math.max(this.wlen[s0], s.length);
        //         this.good[s0].add(s1);
        //     }
        // } Convert to c++
        table = std::vector<std::vector<std::vector<std::vector<uchar>>>>(256, std::vector<std::vector<std::vector<uchar>>>(256, std::vector<std::vector<uchar>>()));
        good = std::vector<std::set<int>>(256, std::set<int>());
        wlen = std::vector<int>(256, 0);

        for (int i = sorted.size() - 1; i >= 0; i--) {
            const std::vector<uchar>& s = sorted[i];
            if (s.size() >= 2) {
                const uchar s0 = s[0];
                const uchar s1 = s[1];
                // init table[s0][s1] if it doesn't exist
                
                table[s0][s1].push_back(s);
                wlen[s0] = std::max(wlen[s0], (int)s.size());
                good[s0].insert(s1);
            }
        }

        // More initializer code would go here to replicate the JavaScript behavior
    }

    // More methods to replicate the JavaScript behavior would go here

    // Sample print function based on printTokens
    void printTokens(const std::vector<size_t>& tokens) {
        for (auto i : tokens) {
            // try {
                std::vector<uchar> s = idx2token[i];
                std::string str(s.begin(), s.end());
                std::cout << "\"" << str << "\"" << i << " ";
            // } catch (...) {
            //     // If the conversion to string fails, keep it in some other format.
            //     // Note: Better error handling is needed
            //     std::cout << "Error" << i << " ";
            // }
        }
        std::cout << std::endl;
    }


    std::vector<size_t> encode(const std::string &src) {
        std::vector<uchar> srcBytes(src.begin(), src.end());
        return encodeBytes(srcBytes);
    }

    std::string decode(const std::vector<size_t> &tokens) {
        std::vector<uchar> byteResult = decodeBytes(tokens);
        return std::string(byteResult.begin(), byteResult.end());
    }

    static bool startsWith(const std::vector<uchar> &target, const std::vector<uchar> &prefix) {
        if (prefix.size() > target.size()) {
            return false;
        }
        return std::equal(prefix.begin(), prefix.end(), target.begin());
    }

private:
    std::vector<size_t> encodeBytes(const std::vector<uchar>& src) {
        const size_t srcLen = src.size();
        std::vector<size_t> tokens;
        size_t i = 0;
        while (i < srcLen) {
            std::vector<uchar> s(src.begin() + i, src.begin() + i + 1);
            if (i < srcLen - 1) {
                const size_t s1 = src[i + 1];
                const size_t s0 = src[i];
                if (good[s0].find(s1) != good[s0].end()) {
                    std::vector<uchar> sss(src.begin() + i, src.begin() + i + wlen[s0]);
                    auto matchIt = std::find_if(table[s0][s1].begin(), table[s0][s1].end(),
                        [&sss](const std::vector<uchar>& t) { return startsWith(sss, t); });
                    if (matchIt != table[s0][s1].end()) {
                        s = *matchIt;
                    }
                }
            }
            std::string sStr(s.begin(), s.end());
            tokens.push_back(token2idx[sStr]);
            i += s.size();
        }
        return tokens;
    }

    std::vector<uchar> decodeBytes(const std::vector<size_t> &tokens) {
        std::vector<uchar> decoded;
        for (size_t token : tokens) {
            const std::vector<uchar> &tokenBytes = idx2token.at(token);
            decoded.insert(decoded.end(), tokenBytes.begin(), tokenBytes.end());
        }
        return decoded;
    }

    

};


// size_t main() {
    
//     std::vector<size_t> sampleTokens = worldTokenizer.encode("Hello World!");
//     std::cout << "Encoded tokens: " << sampleTokens[0] << " " << sampleTokens[1] << std::endl;
//     std::cout << worldTokenizer.decode({33155, 37576}) << std::endl;
//     return 0;
// }

#endif // BYTEMATCH_HPP