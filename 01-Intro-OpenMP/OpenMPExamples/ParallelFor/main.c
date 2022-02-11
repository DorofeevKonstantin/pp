#include <stdio.h>
#include <omp.h>

#define SIZE 10

int main()
{
	int a[SIZE], b[SIZE], c[SIZE];
	omp_set_num_threads(4);
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < SIZE; i++)
		c[i] = a[i] + b[i];
	return 0;
}