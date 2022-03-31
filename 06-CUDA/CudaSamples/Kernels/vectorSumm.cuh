#include "cuda_runtime.h"

__global__ void vectorSummBlocksKernel(int* c, int* a, int* b, const int size);

void printVectorSumm(const int* a, const int* b, const int* c, const int size);