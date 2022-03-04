#include <iostream>
#include <Windows.h>

#include "mpi.h"

int main(int argc, char** argv)
{
	int myRank, numProcs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	int length = 0;
	char name[100];
	MPI_Get_processor_name(name, &length);
	std::cout << "Process " << myRank << " of " << numProcs << " is running on " << name << std::endl;
	MPI_Finalize();
    return 0;
}