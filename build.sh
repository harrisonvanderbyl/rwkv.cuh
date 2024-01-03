
export HEADERFILES=$(locate cuda_runtime.h | sed 's/\/cuda_runtime.h//')
export LIBRARYFILES=$(locate libcudart_static.a | sed 's/\/libcudart_static.a//')

# ensure only first line is used
export LIBRARYFILES=$(echo $LIBRARYFILES | sed 's/ .*//')
export HEADERFILES=$(echo $HEADERFILES | sed 's/ .*//')
# add ./include to header files
export HEADERFILES="$HEADERFILES -I./include"

# include debug symbols
export DEBUG="-g -G"


g++ ./cpuops.cpp -I./include/ -pthread -o ./cpuops.o -c -g 

nvcc ./rwkv.cu ./cpuops.o  -I$HEADERFILES -L$LIBRARYFILES $DEBUG  -o ./rwkv 
# g++ ./rwkv.cpp ./cpuops.o  -I./include -o ./rwkvcpu -march=native -pthread