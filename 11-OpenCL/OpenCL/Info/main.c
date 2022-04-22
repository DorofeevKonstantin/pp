#include <stdio.h>

#include <CL/cl.h>

int main(int argc, char* argv[])
{
	char plat_name[50], dev_name[50], plat_version[50], dev_version[50];
	int choise = 0, error = 0;
	const int int_value = 15;
	cl_uint num_platforms = 0, num_devices = 0;
	cl_ulong mem_size;
	cl_device_type dev_type;
	cl_platform_id platforms[10];
	cl_device_id devices[10];
	cl_platform_info platinfo = 0;
	clGetPlatformIDs(10, platforms, &num_platforms);
	printf_s("Find %d OpenCL platforms on this system.\n", num_platforms);
	for (size_t i = 0; i < num_platforms; i++)
	{
		printf_s("\nPlatform %d info:\n", i);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 50 * sizeof(char), plat_name, NULL);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 50 * sizeof(char), plat_version, NULL);
		printf_s("Name: %s Version: %s\n", plat_name, plat_version);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
		printf_s("Find %d devices.\n", num_devices);
		for (unsigned int j = 0; j < num_devices; j++)
		{
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 50 * sizeof(char), dev_name, NULL);
			printf_s("Device name: %s\n", dev_name);
			clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
			if (dev_type == (1 << 2))
				printf_s("Device type: GPU\n");
			if (dev_type == (1 << 1))
				printf_s("Device type: CPU\n");
			clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
			printf_s("Device global_mem_size: %lld MB\n", mem_size / (1024 * 1024));
			clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
			printf_s("Device local_mem_size: %lld KB\n", mem_size / 1024);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 50 * sizeof(char), dev_version, NULL);
			printf_s("Device version: %s\n", dev_version);
		}
	}
	return 0;
}