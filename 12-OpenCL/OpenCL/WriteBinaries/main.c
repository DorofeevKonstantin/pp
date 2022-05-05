#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

int readFile(char** output, size_t* size, const char* name)
{
	FILE* fp = fopen(name, "rb");
	if (!fp)
		return -1;
	fseek(fp, 0, SEEK_END);
	*size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	*output = (char*)malloc(*size);
	if (!*output)
	{
		fclose(fp);
		return -1;
	}
	fread(*output, *size, 1, fp);
	fclose(fp);
	return 0;
}
int writeFile(const char* name, const unsigned char* content, size_t size)
{
	FILE* fp = fopen(name, "wb+");
	if (!fp)
		return -1;
	fwrite(content, size, 1, fp);
	fclose(fp);
	return 0;
}
cl_int getPlatformList(cl_platform_id** platforms_out, cl_uint* num_platforms_out)
{
	cl_int err;
	cl_uint num_platforms;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	if (err != CL_SUCCESS)
		return err;
	if (num_platforms == 0)
		return CL_INVALID_VALUE;
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	if (!platforms)
		return CL_OUT_OF_HOST_MEMORY;
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		free(platforms);
		return err;
	}
	*platforms_out = platforms;
	*num_platforms_out = num_platforms;
	return CL_SUCCESS;
}
void freePlatformList(cl_platform_id* platforms, cl_uint num_platforms)
{
	free(platforms);
}
char* getPlatformInfo(cl_platform_id platform, cl_platform_info param)
{
	cl_int err;
	size_t buf_size;
	err = clGetPlatformInfo(platform, param, 0, NULL, &buf_size);
	if (err != CL_SUCCESS)
		return NULL;
	if (buf_size == 0)
		return NULL;
	char* buf = (char*)malloc(buf_size);
	if (!buf)
		return NULL;
	err = clGetPlatformInfo(platform, param, buf_size, buf, NULL);
	if (err != CL_SUCCESS)
	{
		free(buf);
		return NULL;
	}
	return buf;
}
cl_int getDeviceList(cl_device_id** devices_out, cl_uint* num_devices_out, cl_platform_id platform)
{
	cl_int err;
	cl_uint num_devices;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	if (err != CL_SUCCESS)
		return err;
	cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	if (!devices)
		return CL_OUT_OF_HOST_MEMORY;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	if (err != CL_SUCCESS)
	{
		free(devices);
		return err;
	}
	*devices_out = devices;
	*num_devices_out = num_devices;
	return CL_SUCCESS;
}
void freeDeviceList(cl_device_id* devices, cl_uint num_devices)
{
	cl_uint i;
	for (i = 0; i < num_devices; ++i)
		clReleaseDevice(devices[i]);
	free(devices);
}
cl_int writeBinaries(cl_program program, unsigned num_devices, cl_uint platform_idx)
{
	unsigned i;
	cl_int err = CL_SUCCESS;
	size_t* binaries_size = NULL;
	unsigned char** binaries_ptr = NULL;
	// Read the binaries size
	size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
	binaries_size = (size_t*)malloc(binaries_size_alloc_size);
	if (!binaries_size)
	{
		err = CL_OUT_OF_HOST_MEMORY;
		goto cleanup;
	}
	err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, binaries_size_alloc_size, binaries_size, NULL);
	if (err != CL_SUCCESS)
		goto cleanup;
	size_t binaries_ptr_alloc_size = sizeof(unsigned char*) * num_devices;
	binaries_ptr = (unsigned char**)malloc(binaries_ptr_alloc_size);
	if (!binaries_ptr)
	{
		err = CL_OUT_OF_HOST_MEMORY;
		goto cleanup;
	}
	memset(binaries_ptr, 0, binaries_ptr_alloc_size);
	for (i = 0; i < num_devices; ++i)
	{
		binaries_ptr[i] = (unsigned char*)malloc(binaries_size[i]);
		if (!binaries_ptr[i])
		{
			err = CL_OUT_OF_HOST_MEMORY;
			goto cleanup;
		}
	}
	err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_ptr_alloc_size, binaries_ptr, NULL);
	if (err != CL_SUCCESS)
		goto cleanup;
	for (i = 0; i < num_devices; ++i)
	{
		char filename[128];
		snprintf(filename, sizeof(filename), "cl-out_%u-%u.bin", (unsigned)platform_idx, (unsigned)i);
		writeFile(filename, binaries_ptr[i], binaries_size[i]);
	}
cleanup:
	if (binaries_ptr)
	{
		for (i = 0; i < num_devices; ++i)
			free(binaries_ptr[i]);
		free(binaries_ptr);
	}
	free(binaries_size);
	return err;
}
cl_int compileProgram(cl_uint* num_devices_out, const char* src, size_t src_size, cl_platform_id platform, cl_uint platform_idx)
{
	cl_int err = CL_SUCCESS;
	cl_device_id* devices = NULL;
	cl_uint num_devices = 0;
	getDeviceList(&devices, &num_devices, platform);
	*num_devices_out = num_devices;
	cl_context_properties ctx_properties[] =
	{
	  CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};
	cl_context ctx = clCreateContext(ctx_properties, num_devices, devices, NULL, NULL, &err);
	if (err != CL_SUCCESS)
		goto cleanup;
	cl_program program = clCreateProgramWithSource(ctx, 1, &src, &src_size, &err);
	if (err != CL_SUCCESS)
		goto cleanup;
	err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
		goto cleanup_program;
	writeBinaries(program, num_devices, platform_idx);
cleanup_program:
	clReleaseProgram(program);
cleanup:
	freeDeviceList(devices, num_devices);
	return err;
}
void compileAll(const char* src, size_t src_size)
{
	cl_uint i;
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms = 0;
	if (getPlatformList(&platforms, &num_platforms) != CL_SUCCESS)
		return;
	for (i = 0; i < num_platforms; ++i)
	{
		cl_uint num_devices = 0;
		cl_int err = compileProgram(&num_devices, src, src_size, platforms[i], i);
		char* platform_name = getPlatformInfo(platforms[i], CL_PLATFORM_NAME);
		printf("PLATFORM [%s]  -->  %s (%u)\n",
			(platform_name ? platform_name : ""),
			((err == CL_SUCCESS) ? "SUCCESS" : "FAILURE"), i);
		fflush(stdout);
		free(platform_name);
	}
	freePlatformList(platforms, num_platforms);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "USAGE: writeBinaries [SOURCE]\n");
		exit(EXIT_FAILURE);
	}
	const char* filename = argv[1];
	char* src = NULL;
	size_t src_size = 0;
	if (readFile(&src, &src_size, filename) != 0)
	{
		fprintf(stderr, "ERROR: Failed to read: %s\n", filename);
		exit(EXIT_FAILURE);
	}
	compileAll(src, src_size);
	free(src);
	return 0;
}