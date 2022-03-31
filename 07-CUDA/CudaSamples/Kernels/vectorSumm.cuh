#include "cuda_runtime.h"

__global__ void vectorSummThreadsKernel(int* a, int* b, int* c, int size);
__global__ void vectorSummLongKernel(int* a, int* b, int* c, int size);

void printVectorSumm(const int* a, const int* b, const int* c, const int size);