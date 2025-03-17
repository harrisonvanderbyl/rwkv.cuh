
export HEADERFILES=$(locate cuda_runtime.h | sed 's/\/cuda_runtime.h//')
export LIBRARYFILES=$(locate libcudart_static.a | sed 's/\/libcudart_static.a//')
nvcc=`locate nvcc | grep -o ".*bin/nvcc" | tail -n 1`
# ensure only first line is used
export LIBRARYFILES=$(echo $LIBRARYFILES | sed 's/ .*//')
export HEADERFILES=$(echo $HEADERFILES | sed 's/ .*//')
# add ./include to header files
export HEADERFILES="$HEADERFILES -I./include"

# include debug symbols
# export DEBUG="-g -G"

# include release symbols
export RELEASE="-O3 --forward-unknown-to-host-compiler --use_fast_math -march=native"

# compile with cuda
# $nvcc ./rwkv.cu ./src/cpuops.cpp -I$HEADERFILES -L$LIBRARYFILES $DEBUG $RELEASE -o ./rwkv.out -arch=sm_80
# $nvcc ./rwkv.cpp ./src/cudaops.cu -I$HEADERFILES -L$LIBRARYFILES $DEBUG $RELEASE -o ./rwkv2.out -arch=sm_80 -g

# compile cpu only
# g++ -x c++ ./rwkv.cu -I./include -o ./rwkvcpu.out -pthread -std=c++17 -march=native -O3 -ffast-math #-ggdb -pg

# compile with hipcc
g++ ./src/cpuops.cpp -c -I./include -o ./cpuops.o -std=c++17 -O3 -ffast-math -march=native -g
hipcc ./rwkv.cpp -c -I./include -o ./rwkv.o -std=c++17 -O3 -ffast-math -march=native -g

# link the object file
hipcc ./rwkv.o ./cpuops.o -o ./rwkv.out -pthread -std=c++17 -march=native -O3 -ffast-math -g

# g++ -x c++ ./testing.cpp ./src/cpuops.cpp  -I./include -o ./tests.out -pthread -std=c++17 -march=native -O3 -ffast-math 