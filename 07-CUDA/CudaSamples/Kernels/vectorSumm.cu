#include <stdio.h>

#include "vectorSumm.cuh"

__global__ void vectorSummThreadsKernel(int* a, int* b, int* c, int size)
{
	int tid = threadIdx.x;
	if (tid < size)
		c[tid] = a[tid] + b[tid];
}
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#control-flow-instructions
__global__ void vectorSummLongKernel(int* a, int* b, int* c, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
void printVectorSumm(const int* a, const int* b, const int* c, const int size)
{
	for (int i = 0; i < size; i++)
		printf("a[%d]:%d + b[%d]:%d == c[%d]:%d\n", i, a[i], i, b[i], i, c[i]);
	printf("\n");
}