
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
export RELEASE="-O3 --use_fast_math --forward-unknown-to-host-compiler -ffast-math -Xptxas -O3 -march=native"

$nvcc -x cu ./rwkv.cpp  -I$HEADERFILES -L$LIBRARYFILES $DEBUG $RELEASE -o ./rwkv.out -arch=sm_80 -g
g++ -x c++ ./rwkv.cpp -I./include -o ./rwkvcpu.out -pthread -std=c++17 -march=native -O3 -ffast-math #-ggdb -pg
# g++ -x c++ ./testing.cpp ./src/cpuops.cpp  -I./include -o ./tests.out -pthread -std=c++17 -march=native -O3 -ffast-math 