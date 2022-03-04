#include <stdio.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <mpi.h>

long long num_steps = 1000000000;

int main(int argc, char **argv)
{
	clock_t start, stop;
	double mypi, sumpi, step, sum = 0.0, x;
	int i;
	int mpirank, mpisize;
	MPI_Init(&argc, &argv);
	start = clock();
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	step = 1. / (double)num_steps;
#pragma omp parallel num_threads(2) default(shared) private(i, x) reduction(+: sum)
	{
		if (mpirank == 0) 
		{
#pragma omp master
			printf_s("Calculating PI with %d processes and %d threads\n", mpisize, omp_get_num_threads());
		}
#pragma omp for
		for (i = mpirank + 1; i <= num_steps; i += mpisize) 
		{
			x = (i + .5) * step;
			sum = sum + 4.0 / (1.0 + x*x);
		}
	}
	mypi = step * sum;
	MPI_Reduce(&mypi, &sumpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	stop = clock();
	if (mpirank == 0) 
	{
		printf_s("%f\n", sumpi);
		printf_s("The time to calculate PI was %f seconds\n", ((double)(stop - start) / 1000.0));
	}
	MPI_Finalize();
    return 0;
}