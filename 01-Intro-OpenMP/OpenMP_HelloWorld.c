#include <stdio.h>
#include <omp.h>

int main()
{
	int size, rank;
	//omp_set_num_threads(5);
	#pragma omp parallel //num_threads(5) private(size, rank)
	{
		rank = omp_get_thread_num();
		printf("Hello World from thread = %d\n", rank);
		getchar();
		if (rank == 0)
		{
			size = omp_get_num_threads();
			printf("Number of threads = %d\n", size);
		}
	}
    return 0;
}