#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

int main(int argc, char** argv)
{
	int size, rank, N = 5;
	int* mass = (int*)malloc(N * sizeof(int));
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0)
	{
		srand((unsigned)time(NULL));
		printf_s("process 0 initialize mass: ");
		for (int i = 0; i < N; i++)
		{
			mass[i] = rand() % 100;
			printf_s("%d ", mass[i]);
		}
		printf_s("and send using MPI_Bcast\n");
	}
	MPI_Bcast(mass, N, MPI_INT, 0, MPI_COMM_WORLD);
	printf_s("process %d receive: ", rank);
	for (int i = 0; i < N; i++)
		printf_s("%d ", mass[i]);
	MPI_Barrier(MPI_COMM_WORLD);
	free(mass);
	MPI_Finalize();
	return 0;
}