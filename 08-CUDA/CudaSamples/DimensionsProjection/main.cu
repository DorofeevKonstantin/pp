#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Error(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
		printf("CUDA error : %s\n", cudaGetErrorString(cudaStatus));
}
//kernel<<<(2,3),(4,5)>>>
//int x = threadIdx.x + blockIdx.x * blockDim.x;
//int y = threadIdx.y + blockIdx.y * blockDim.y;
//thread(3,4) in block(1,2) start job
//there are 1 block to the left
//there are 2 block to the top
//each block contains (4,5) threads
//so globally there are 1*4+3 threads to the left
//					and 2*5+4 threads to the top
__global__ void _2dProjectionKernel()
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d,%d\n", row, col);
}
void _2dTaskDimensionProjectionExample()
{
	// NxN may present any 2d task (e.g. matrix addition)
	const int N = 4;
	const int BLOCK_SIZE = 2;
	dim3 grid2D(N / BLOCK_SIZE, N / BLOCK_SIZE);
	dim3 block2D(BLOCK_SIZE, BLOCK_SIZE);
	_2dProjectionKernel << <grid2D, block2D >> > ();
	Error(cudaGetLastError());
	Error(cudaDeviceSynchronize());
}

int main()
{
	_2dTaskDimensionProjectionExample();
	return 0;
}