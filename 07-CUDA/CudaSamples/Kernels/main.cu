#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vectorSumm.cuh"

#define SIZE 10

void vectorSummExample()
{
	int a[SIZE], b[SIZE], c[SIZE];
	int* dev_a, * dev_b, * dev_c;
	cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
	for (int i = 0; i < SIZE; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}
	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	vectorSummThreadsKernel << <1, SIZE >> > (dev_a, dev_b, dev_c, SIZE);
	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printVectorSumm(a, b, c, SIZE);

	vectorSummLongKernel << <2, 2 >> > (dev_a, dev_b, dev_c, SIZE);
	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printVectorSumm(a, b, c, SIZE);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

int main()
{
	vectorSummExample();
	return 0;
}