#include <iostream>

#include "CL\cl.hpp"

int main(int argc, char* argv[])
{
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0)
	{
		std::cout << "No platforms found. Check OpenCL installation." << std::endl;
		exit(-1);
	}
	std::cout << "Find " << all_platforms.size() << " OpenCL platforms on this system." << std::endl;
	for (unsigned int i = 0; i < all_platforms.size(); i++)
	{
		cl::Platform default_platform = all_platforms[i];
		std::cout << std::endl << "Platform name: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::cout << "Platform version: " << default_platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		if (all_devices.empty())
		{
			std::cout << "No devices found. Check OpenCL installation!" << std::endl;
			exit(-1);
		}
		std::cout << "Find " << all_devices.size() << " devices on this platform." << std::endl;
		for (size_t j = 0; j < all_devices.size(); j++)
		{
			cl::Device default_device = all_devices[j];
			std::cout << "Device name: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
			std::cout << "Device type: " << default_device.getInfo<CL_DEVICE_TYPE>() << std::endl;
			std::cout << "Device global_mem_size: " << default_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << std::endl;
			std::cout << "Device local_mem_size: " << default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / (1024) << std::endl;
			std::cout << "Device version: " << default_device.getInfo<CL_DEVICE_VERSION>() << "\n";
		}
	}
	return 0;
}