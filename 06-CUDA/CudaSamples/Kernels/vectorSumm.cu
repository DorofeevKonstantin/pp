#include <stdio.h>

#include "vectorSumm.cuh"

__global__ void vectorSummBlocksKernel(int* c, int* a, int* b, const int size)
{
	int i = blockIdx.x;
	if (i < size)
		c[i] = a[i] + b[i];
}
void printVectorSumm(const int* a, const int* b, const int* c, const int size)
{
	for (int i = 0; i < size; i++)
		printf("a[%d]:%d + b[%d]:%d == c[%d]:%d\n", i, a[i], i, b[i], i, c[i]);
}