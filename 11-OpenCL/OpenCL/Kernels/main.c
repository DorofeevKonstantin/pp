#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl.h"

char* readKernelFile(const char* filename, long* _size)
{
	FILE* file;
	fopen_s(&file, filename, "r");
	if (!file) {
		printf("-- Error opening file %s\n", filename);
		exit(-1);
	}
	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	rewind(file);
	// Read the kernel code as a string
	char* source = (char*)malloc((size + 1) * sizeof(char));
	fread(source, 1, size * sizeof(char), file);
	source[size] = '\0';
	fclose(file);
	// Save the size and return the source string
	*_size = (size + 1);
	return source;
}
// Set the kernel as a string (better to do this in a separate file though)
const char* kernelstring =
"__kernel void simple_kernel(const int value)	"
"    {											"
"    const int row = get_global_id(0);			"
"    const int col = get_global_id(1);			"
"    const int depth = get_global_id(2);		"
"    const int lRow = get_local_id(0);			"
"    const int lCol = get_local_id(1);			"
"    const int lDepth = get_local_id(2);		"
"    int result = value*(row+col);				"
"	 printf(\"kernel %d,%d,%d %d,%d,%d value=%d -> result = %d\\n\", row,col,depth,lRow,lCol,lDepth,value,result);"
"    }";

int main(int argc, char* argv[])
{
	char plat_name[50], dev_name[50], plat_version[50], dev_version[50];
	int choise = 0, error = 0, gpu_device_index = -1;
	cl_uint num_platforms = 0, num_devices = 0;
	cl_ulong global_mem_size;
	cl_device_type dev_type;
	cl_platform_id platforms[10];
	cl_device_id devices[10];
	cl_platform_info platinfo = 0;
	clGetPlatformIDs(10, platforms, &num_platforms);
	for (unsigned int i = 0; i < num_platforms; i++)
	{
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 50 * sizeof(char), plat_name, NULL);
		if (strstr(plat_name, "NVIDIA") == NULL)
			continue;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
		for (unsigned int j = 0; j < num_devices; j++)
		{
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 50 * sizeof(char), dev_name, NULL);
			if (strstr(dev_name, "NVIDIA") != NULL)
			{
				gpu_device_index = j;
				break;
			}
		}
	}
	if (gpu_device_index == -1)
		exit(-1);
	// Read the kernel file from disk
	/*
	long sizeSource;
	char* source = readKernelFile("kernels.cl", &sizeSource);
	const char* constCode = source;
	free(source);
	*/
	const char* constCode = kernelstring;
	cl_context context = clCreateContext(NULL, 1, &devices[gpu_device_index], NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, devices[gpu_device_index], 0, NULL);
	cl_event event = NULL;
	cl_program program = clCreateProgramWithSource(context, 1, &constCode, NULL, NULL);
	clBuildProgram(program, 0, NULL, "", NULL, NULL);
	size_t logSize;
	cl_int buildResult = clGetProgramBuildInfo(program, devices[gpu_device_index], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	char* logInfo = (char*)malloc((1 + logSize) * sizeof(char));
	buildResult = clGetProgramBuildInfo(program, devices[gpu_device_index], CL_PROGRAM_BUILD_LOG, logSize, logInfo, NULL);
	logInfo[logSize] = '\0';
	if (buildResult != CL_SUCCESS)
		printf("Error build info:\n%s\n", logInfo);
	free(logInfo);
	//cl_mem value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
	//clEnqueueWriteBuffer(queue, value, CL_TRUE, 0, sizeof(int), &int_value, 0, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "simple_kernel", NULL);
	int int_value = 15;
	clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&int_value);
	printf("Starting simple_kernel.\n");
	size_t global[] = { 4,4 };
	size_t local[] = { 2,2 };
	cl_int result = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
	clWaitForEvents(1, &event);
	//clReleaseMemObject(value);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}