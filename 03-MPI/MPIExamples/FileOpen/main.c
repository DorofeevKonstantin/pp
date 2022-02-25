#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

int main()
{
	MPI_Init(0, 0);
	int bufsize, count, rank, size, tmp;
	unsigned char* buf;
	MPI_Status status;
	MPI_File file;
	MPI_Offset filesize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_File_open(MPI_COMM_WORLD, "test.txt", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	MPI_File_get_size(file, &filesize);  // in bytes 
	filesize = filesize / sizeof(unsigned char);    // in number of uchars 
	if (rank == size - 1)
	{
		//printf_s("Filesize = %lld\n", filesize);
		tmp = (int)filesize / size;
		bufsize = (int)filesize - (size - 1) * tmp;
	}
	else
		bufsize = (int)filesize / size;

	printf_s("Bufsize = %d in %d process. ", bufsize, rank);
	buf = (unsigned char*)malloc((bufsize + 1) * sizeof(unsigned char));
	MPI_File_set_view(file, rank * (filesize / size) * sizeof(unsigned char), MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);
	MPI_File_read(file, buf, bufsize, MPI_UNSIGNED_CHAR, &status);
	MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
	buf[bufsize] = '\0';
	printf_s("Process %d read %d chars : %s\n", rank, count, buf);
	MPI_File_close(&file);
	MPI_Finalize();
	return 0;
}