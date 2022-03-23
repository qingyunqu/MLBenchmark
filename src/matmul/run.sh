nvcc -o cublas matmul.cpp -std=c++17 -I/root/share/MLBenchmark/include -lcublas
nvcc -o cutlass matmul.cu  -std=c++17 -I/root/share/MLBenchmark/include -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include -arch=sm_75 -lcublas

nvcc -o gemm gemm.cu  -std=c++17 -I/root/share/MLBenchmark/include -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include -arch=sm_75 -lcublas