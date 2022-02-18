#include <iostream>

#include "mpi.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int myRank, numProcs;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	if (myRank == 0)
	{
		int sendValue = 15;
		MPI_Send(&sendValue, 1, MPI_INT, 1, 170, MPI_COMM_WORLD);
		std::cout << myRank << " send " << sendValue << std::endl;
	}
	else if (myRank == 1)
	{
		MPI_Status status;
		int recvValue;
		MPI_Recv(&recvValue, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		std::cout << myRank << " recv " << recvValue << " from " << status.MPI_SOURCE
			<< " tag " << status.MPI_TAG << std::endl;
	}
	MPI_Finalize();
	return 0;
}