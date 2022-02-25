#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int* sendBuf = (int*)malloc((size * 2) * sizeof(int));
	int* recvBuf = (int*)malloc(2 * sizeof(int));
	if (rank == 0)
	{
		srand((unsigned)time(NULL) + rank);
		int correctSumm = 0;
		printf_s("Process 0 generate mass : ");
		for (int i = 0; i < size * 2; i++)
		{
			sendBuf[i] = rand() % 200 - 100;
			correctSumm += sendBuf[i];
			printf_s("%d ", sendBuf[i]);
		}
		printf_s(" correctSumm = %d\n", correctSumm);
	}
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatter(sendBuf, 2, MPI_INT, recvBuf, 2, MPI_INT, 0, MPI_COMM_WORLD);
	int partialSumm = recvBuf[0] + recvBuf[1];
	printf_s("(Scatter) process %d receive <-- %d,%d, sending ->  %d summ", rank, recvBuf[0], recvBuf[1], partialSumm);
	int* recvPartialSumms = (int*)malloc(size * sizeof(int));
	if (rank == 0)
	{
		for (int i = 0; i < size; i++)
			recvPartialSumms[i] = -999;
	}
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(&partialSumm, 1, MPI_INT, recvPartialSumms, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		int potentialSumm = 0;
		printf_s("(Gather) process 0 receive using Gather: ");
		for (int i = 0; i < size; i++)
		{
			printf_s("%d ", recvPartialSumms[i]);
			potentialSumm += recvPartialSumms[i];
		}
		printf_s("potentialSumm = %d\n", potentialSumm);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free(recvBuf);
	free(sendBuf);
	MPI_Finalize();
	return 0;
}