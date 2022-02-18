#include <iostream>

#include "mpi.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int myRank, numProcs;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	std::cout << "Hello world!" << std::endl;
	MPI_Finalize();
	return 0;
}