#include <stdio.h>
#include <time.h>
#include <math.h>

#include "mpi.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Error(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("Some Error : %s\n", cudaGetErrorString(cudaStatus));
	}
}
__global__ void kernel()
{
	printf("kernel <<<(%d,%d,%d),(%d,%d,%d)>>>\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}
void DeviceOutput(cudaDeviceProp& prop)
{
	printf("CUDA Device info:\n");
	printf("Name: %s\n", prop.name);
	printf("Architecture: %d.%d\n", prop.major, prop.minor);
	printf("Clockrate: %d\n", prop.clockRate);
	printf("Globalmem: %ld\n", prop.totalGlobalMem);
	printf("SM: %d\n", prop.multiProcessorCount);
	printf("WarpSize: %d\n", prop.warpSize);
	printf("ThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

int main(void)
{
	MPI_Init(0, 0);
	cudaDeviceProp prop;
	int size, rank, length = 0, count;
	char name[100];
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	std::cout << "Process " << rank << " of " << size << " is running on " << name << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&count);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount failed!\n");
		return -1;
	}
	printf("%d CUDA devices was found\n", count);
	for (int i = 0; i < count; i++)
	{
		cudaStatus = cudaGetDeviceProperties(&prop, i);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaGetDeviceProperties failed!\n");
			return -1;
		}
		else
			DeviceOutput(prop);
	}
	kernel << <2, 2 >> > ();
	MPI_Finalize();
	return 0;
}