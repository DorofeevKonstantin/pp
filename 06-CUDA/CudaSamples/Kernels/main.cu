#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vectorSumm.cuh"

__global__ void simpleKernel(void)
{
	printf("i am %d %d %d  block\n", blockIdx.x, blockIdx.y, blockIdx.z);
	if (blockIdx.x == 0)
		printf(" -> threads count in block : %d\n", gridDim.x);
}
void simpleExample()
{
	simpleKernel << <5, 1 >> > ();
	cudaDeviceSynchronize();
	printf("simpleKernel end\n");
}

__global__ void addIntegersKernel(int a, int b, int* c)
{
	*c = a + b;
}
void addIntegersExample()
{
	int c;
	int* dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	addIntegersKernel << <2, 2 >> > (2, 7, dev_c);
	cudaDeviceSynchronize();
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("addIntegersKernel(2,7) == %d\n", c);
	cudaFree(dev_c);
}

void vectorSummExample()
{
	const int size = 10;
	int* a, * b, * c;
	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	c = (int*)malloc(size * sizeof(int));
	int* dev_a, * dev_b, * dev_c;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	for (int i = 0; i < size; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	vectorSummKernel << <size, 1 >> > (dev_c, dev_a, dev_b, size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	printVectorSumm(a, b, c, size);
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	free(a);
	free(b);
	free(c);
}

int main()
{
	simpleExample();
	addIntegersExample();
	vectorSummExample();
	return 0;
}