#include <iostream>
#include <Windows.h>

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
		MPI_Request request;
		MPI_Status status;
		MPI_Isend(&sendValue, 1, MPI_INT, 1, 170, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		std::cout << myRank << " send " << sendValue << std::endl;
	}
	else if (myRank == 1)
	{
		Sleep(5000);
		MPI_Request request;
		MPI_Status status;
		int recvValue;
		MPI_Irecv(&recvValue, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		std::cout << myRank << " recv " << recvValue << " from " << status.MPI_SOURCE
			<< " tag " << status.MPI_TAG << std::endl;
	}
	MPI_Finalize();
	return 0;
}