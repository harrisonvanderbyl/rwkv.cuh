
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

nvcc ./rwkv.cu ./cpuops.cpp  -I$HEADERFILES -L$LIBRARYFILES $DEBUG $RELEASE -o ./rwkv -arch=sm_80 
# g++ ./rwkv.cu.cpp ./cpuops.cpp  -I./include -o ./rwkvcpu -pthread