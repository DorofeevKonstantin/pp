#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int N = 1000;
const int threadsPerBlock = 256;
const int blocksPerGrid = 2;

void Error(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
		printf("CUDA error : %s\n", cudaGetErrorString(cudaStatus));
}
__global__ void scalarMultKernel(float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main(void)
{
	float* a, * b, c, * partialC;
	float* devA, * devB, * devPartialC;
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partialC = (float*)malloc(blocksPerGrid * sizeof(float));
	Error(cudaMalloc((void**)&devA, N * sizeof(float)));
	Error(cudaMalloc((void**)&devB, N * sizeof(float)));
	Error(cudaMalloc((void**)&devPartialC, blocksPerGrid * sizeof(float)));
	for (int i = 0; i < N; i++)
	{
		a[i] = 1;
		b[i] = i;
	}
	Error(cudaMemcpy(devA, a, N * sizeof(float), cudaMemcpyHostToDevice));
	Error(cudaMemcpy(devB, b, N * sizeof(float), cudaMemcpyHostToDevice));
	scalarMultKernel << <blocksPerGrid, threadsPerBlock >> > (devA, devB, devPartialC);
	Error(cudaMemcpy(partialC, devPartialC, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
		c += partialC[i];
	printf("Value = %f\n", c);
	Error(cudaFree(devA));
	Error(cudaFree(devB));
	Error(cudaFree(devPartialC));
	free(a);
	free(b);
	free(partialC);
}