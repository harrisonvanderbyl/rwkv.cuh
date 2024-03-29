
export HEADERFILES=$(locate cuda_runtime.h | sed 's/\/cuda_runtime.h//')
export LIBRARYFILES=$(locate libcudart_static.a | sed 's/\/libcudart_static.a//')

# ensure only first line is used
export LIBRARYFILES=$(echo $LIBRARYFILES | sed 's/ .*//')
export HEADERFILES=$(echo $HEADERFILES | sed 's/ .*//')
# add ./include to header files
export HEADERFILES="$HEADERFILES -I./../include"

# include debug symbols
# nvcc ./tests.cu ./../src/cpuops.cpp -I$HEADERFILES -L$LIBRARYFILES -o ./testsgpu -arch=sm_80 -g
g++ ./testscpu.cpp ./../src/cpuops.cpp -I./../include -o ./cputest.out -std=c++11 -g