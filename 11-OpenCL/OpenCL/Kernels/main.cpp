#include <iostream>

#include "CL/cl.hpp"

std::string kernel_code =
"   __kernel void simple_kernel(const int value, global int* mass)				"
"	{																			"
"		printf(\"Hello from kernel %d\\n\", get_global_id(0));					"
"		printf(\"kernel value = %d\\n\", value);								"
"		printf(\"mass = %d\\n\", mass[get_global_id(0)] );						"
"   }																			";

int main(int argc, char* argv[])
{
	int plat_i = -1, dev_j = -1;
	std::vector<cl::Platform> all_platforms;
	std::vector<cl::Device> all_devices;
	cl::Platform::get(&all_platforms);
	cl::Platform default_platform;
	cl::Device default_device;
	size_t pos;
	for (unsigned int i = 0; i < all_platforms.size(); i++)
	{
		default_platform = all_platforms[i];
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
		for (unsigned int j = 0; j < all_devices.size(); j++)
		{
			default_device = all_devices[j];
			std::string s = default_device.getInfo<CL_DEVICE_NAME>();
			pos = s.find("NVIDIA");
			if (pos != std::string::npos)
			{
				plat_i = i;
				dev_j = j;
				break;
			}
		}
		if (plat_i != -1 && dev_j != -1)
			break;
	}
	if (plat_i == -1 || dev_j == -1)
	{
		std::cout << "Not found CUDA device\n";
		exit(-1);
	}
	else
	{
		std::cout << "Platform: " << plat_i << std::endl;
		std::cout << "Device: " << dev_j << std::endl;
	}
	default_platform = all_platforms[plat_i];
	default_device = all_devices[dev_j];
	std::cout << "Platform name : " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Device name : " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

	cl::Context context(default_device);
	cl::Program::Sources sources;
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });
	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(-1);
	}
	cl::CommandQueue queue(context, default_device);
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 5);
	std::vector<int> h_v = { 10, 11, 12, 13, 14 };
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 5, &h_v[0]);
	cl::Kernel simple_kernel(program, "simple_kernel");
	simple_kernel.setArg(0, 15);
	simple_kernel.setArg(1, buffer_A);
	queue.enqueueNDRangeKernel(simple_kernel, cl::NullRange, cl::NDRange(5), cl::NullRange);
	queue.finish();
	return 0;
}