nvcc -o cublas matmul.cpp -I/root/share/MLBenchmark/include -lcublas -std=c++17
nvcc -o cutlass matmul.cu  -I/root/share/MLBenchmark/include -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include -arch=sm_75 -lcublas

nvcc -o gemm gemm.cu  -I/root/share/MLBenchmark/include -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include -arch=sm_75 -lcublas