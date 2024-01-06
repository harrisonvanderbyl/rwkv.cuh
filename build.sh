
export HEADERFILES=$(locate cuda_runtime.h | sed 's/\/cuda_runtime.h//')
export LIBRARYFILES=$(locate libcudart_static.a | sed 's/\/libcudart_static.a//')

# ensure only first line is used
export LIBRARYFILES=$(echo $LIBRARYFILES | sed 's/ .*//')
export HEADERFILES=$(echo $HEADERFILES | sed 's/ .*//')
# add ./include to header files
export HEADERFILES="$HEADERFILES -I./include"

# include debug symbols
# export DEBUG="-g -G"

# include release symbols
export RELEASE="-O3 --use_fast_math --forward-unknown-to-host-compiler -ffast-math -Xptxas -O3"

nvcc ./rwkv.cu ./src/cpuops.cpp  -I$HEADERFILES -L$LIBRARYFILES $DEBUG $RELEASE -o ./rwkv.out -arch=sm_80 -g
g++ -x c++ ./rwkv.cu ./src/cpuops.cpp  -I./include -o ./rwkvcpu.out -pthread -O3 -ffast-math -std=c++11