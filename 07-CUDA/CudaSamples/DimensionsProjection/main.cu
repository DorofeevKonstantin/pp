#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Error(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
		printf("Some Error : %s\n", cudaGetErrorString(cudaStatus));
}

__global__ void printAllIndexesKernel(void)
{
	printf("%d,%d,%d thread in %d,%d,%d block\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
	if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) && (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0))
		printf("kernel <<<(%d,%d,%d),(%d,%d,%d)>>>\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}
void dimensionOutputExample()
{
	dim3 dimGrid(1, 2, 2);
	dim3 dimBlock(1, 2, 3);
	printAllIndexesKernel << <dimGrid, dimBlock >> > ();
	cudaDeviceSynchronize();
}

__global__ void linearProjectionKernel(int val)
{
	int GDx = gridDim.x, GDy = gridDim.y, GDz = gridDim.z;
	int BDx = blockDim.x, BDy = blockDim.y, BDz = blockDim.z;
	int b_x = blockIdx.x, b_y = blockIdx.y, b_z = blockIdx.z;
	int t_x = threadIdx.x, t_y = threadIdx.y, t_z = threadIdx.z;
	int global_id_thread = t_x + t_y * BDx + t_z * BDx * BDy;
	int global_id_block = b_x + b_y * GDx + b_z * GDx * GDy;
	int index = global_id_block * (BDx * BDy * BDz) + global_id_thread;
	printf("block %d.%d.%d,thread %d.%d.%d id_block=%d id_thread=%d index=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, global_id_block, global_id_thread, index);
}
void linearTaskDimensionProjectionExample()
{
	// N may present any linear task (e.g. vector addition)
	const int N = 16; // need N==(a*2)*(a*2)
	const int BLOCK_SIZE = 2;
	const int n_sqrt = sqrt(N);
	// we may want to configure the kernel as a 2D grid with 2D blocks
	dim3 grid2D(n_sqrt / BLOCK_SIZE, n_sqrt / BLOCK_SIZE);
	dim3 block2D(BLOCK_SIZE, BLOCK_SIZE);
	linearProjectionKernel << <grid2D, block2D >> > (1);
	cudaDeviceSynchronize();
}

int main()
{
	dimensionOutputExample();
	linearTaskDimensionProjectionExample();
	return 0;
}