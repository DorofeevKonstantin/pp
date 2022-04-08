// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZE 10

__device__ __managed__ int ret[SIZE];

__global__ void AplusB(int a, int b)
{
	ret[threadIdx.x] = a + b + threadIdx.x;
}

int main()
{
	AplusB << < 1, SIZE >> > (1, 2);
	cudaDeviceSynchronize();
	for (int i = 0; i < SIZE; ++i)
		printf("%d: A+B+index == %d\n", i, ret[i]);
	return 0;
}