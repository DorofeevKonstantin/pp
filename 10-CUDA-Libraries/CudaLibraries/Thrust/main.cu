#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include <thrust/detail/vector_base.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/random.h>

#define SIZE 100000

struct deviceSummFunctor
{
	deviceSummFunctor() {}
	__device__ float operator()(const float& x, const float& y) const
	{
		return x + y;
	}
};
// this couldn't work
//__device__ float deviceRand()
//{
//	return ((float)rand() / RAND_MAX);
//}
struct devRand
{
	__device__
		float operator () (int idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_real_distribution<float> uniDist;
		randEng.discard(idx);
		return uniDist(randEng);
	}
};
void summ_fast(thrust::device_vector<float>& X, thrust::device_vector<float>& Y, thrust::device_vector<float>& Z)
{
	//thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), thrust::multiplies<float>());
	thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), deviceSummFunctor());
}
void example1()
{
	std::cout << "example1" << std::endl;
	thrust::host_vector<int> h_vec(3);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);
	for (size_t i = 0; i < h_vec.size(); ++i)
		std::cout << "Generate Host_vector [" << i << "] = " << h_vec[i] << std::endl;
	thrust::device_vector<int> d_vec = h_vec;
	thrust::sort(d_vec.begin(), d_vec.end());
	//thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	for (size_t i = 0; i < d_vec.size(); ++i)
		std::cout << "Sorted Device_vector [" << i << "] = " << d_vec[i] << std::endl;
	d_vec[0] = 9999;
	for (size_t i = 0; i < d_vec.size(); ++i)
		std::cout << "Modified Device_vector [" << i << "] = " << d_vec[i] << std::endl;
}
void example2()
{
	std::cout << "example2" << std::endl;
	std::list<int> stl_list;
	stl_list.push_back(10);
	stl_list.push_back(20);
	stl_list.push_back(30);
	thrust::device_vector<int> D(stl_list.begin(), stl_list.end());
	for (size_t i = 0; i < D.size(); i++)
		std::cout << "From list:Device_vector[" << i << "] = " << D[i] << std::endl;
	std::vector<int> stl_vector(D.size());
	thrust::copy(D.begin(), D.end(), stl_vector.begin());
	for (size_t i = 0; i < D.size(); i++)
		std::cout << "STL Vector from Device_vector[" << i << "] = " << D[i] << std::endl;
}
void hard_job_device()
{
	std::cout << "HARD JOB DEVICE" << std::endl;
	clock_t start = clock();

	thrust::host_vector<float> h_a(SIZE);
	thrust::host_vector<float> h_b(SIZE);
	thrust::generate(h_a.begin(), h_a.end(), rand);
	thrust::generate(h_b.begin(), h_b.end(), rand);
	thrust::device_vector<float> d_a = h_a;
	thrust::device_vector<float> d_b = h_b;
	thrust::device_vector<float> d_c(SIZE);

	/*thrust::device_vector<float> d_a(SIZE);
	thrust::device_vector<float> d_b(SIZE);
	thrust::device_vector<float> d_c(SIZE);
	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(SIZE),
		d_a.begin(),
		devRand());
	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(SIZE),
		d_b.begin(),
		devRand());*/

	for (size_t i = 0; i < SIZE; i++)
		summ_fast(d_a, d_b, d_c);
	clock_t end = clock() - start;
	std::cout << " thrust summ " << (float)end / CLOCKS_PER_SEC << " sec" << std::endl;
	if (SIZE < 10)
	{
		for (unsigned int i = 0; i < SIZE; i++)
			std::cout << "RESULT:" << std::endl << "D_c[" << i << "] = " << d_c[i] << std::endl;
	}
}
template <class T>
void printStdVector(const std::vector<T>& v, const std::string& message)
{
	if (v.size() > 10)
		return;
	for (size_t i = 0; i < v.size(); i++)
		std::cout << message << "[" << i << "] = " << v[i] << std::endl;
}
void hard_job_cpu()
{
	std::cout << "HARD JOB CPU" << std::endl;
	clock_t start = clock();
	std::vector<float> a(SIZE);
	std::vector<float> b(SIZE);
	std::vector<float> c(SIZE);
	for (unsigned int i = 0; i < SIZE; i++)
	{
		a[i] = rand() % SIZE;
		b[i] = rand() % SIZE;
	}
	printStdVector(a, "Hard job a");
	printStdVector(b, "Hard job b");
	for (unsigned int j = 0; j < SIZE; j++)
	{
		std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<float>());
	}
	clock_t end = clock() - start;
	std::cout << " STL summ " << (float)end / CLOCKS_PER_SEC << " sec" << std::endl;
	printStdVector(c, "Hard job c");
}

// more examples https://github.com/NVIDIA/thrust/tree/main/examples
int main(void)
{
	example1();
	example2();
	hard_job_device();
	hard_job_cpu();
	return 0;
}