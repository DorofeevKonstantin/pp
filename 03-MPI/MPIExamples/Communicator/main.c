#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int size, sizeOdd, rank, groupRank = -1, rankOdd = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Group groupWorld, oddGroup;
	MPI_Comm commEven;
	int i, neven;
	int* members = (int*)malloc((size / 2) * sizeof(int));
	MPI_Comm_group(MPI_COMM_WORLD, &groupWorld);
	neven = (size + 1) / 2;
	for (i = 0; i < neven; i++)
		members[i] = 2 * i;
	MPI_Group_incl(groupWorld, neven, members, &oddGroup);
	MPI_Comm_create(MPI_COMM_WORLD, oddGroup, &commEven);
	//MPI_Group_rank(odd_group, &group_rank);
	if (commEven != MPI_COMM_NULL)
	{
		MPI_Comm_rank(commEven, &rankOdd);
		MPI_Comm_size(commEven, &sizeOdd);
		printf_s("i am %d from %d in NEW_COMM_ODD and %d in COMM_WORLD\n", rankOdd, sizeOdd, rank);
		MPI_Group_free(&oddGroup);
		MPI_Comm_free(&commEven);
	}
	MPI_Finalize();
	return 0;
}