#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void printDeviceProperties(cudaDeviceProp* prop)
{
	printf("CUDA Device info:\n");
	printf("Name: %s\n", prop->name);
	printf("Architecture: %d.%d\n", prop->major, prop->minor);
	printf("Clockrate: %d Mhz\n", prop->clockRate / 1000);
	printf("Globalmem: %zd Mbytes\n", prop->totalGlobalMem / (1024 * 1024));
	printf("SharedmemPerBlock: %zd Kbytes\n", prop->sharedMemPerBlock / 1024);
	printf("multiProcessors: %d\n", prop->multiProcessorCount);
	printf("WarpSize: %d\n", prop->warpSize);
	printf("ThreadsPerBlock: %d\n", prop->maxThreadsPerBlock);
	printf("Maximum Grid Size: (%d, %d, %d)\n", prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
	printf("Maximum Block Size: (%d, %d, %d)\n", prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
}

int main()
{
	cudaDeviceProp deviceProperties;
	int devicesCount;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&devicesCount);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount failed!\n");
		return -1;
	}
	printf("%d devices was found\n", devicesCount);
	for (int i = 0; i < devicesCount; i++)
	{
		cudaStatus = cudaGetDeviceProperties(&deviceProperties, i);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaGetDeviceProperties failed!\n");
			return -1;
		}
		else
			printDeviceProperties(&deviceProperties);
	}
	return 0;
}