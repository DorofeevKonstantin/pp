#include <stdio.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

void Error(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("Some Error : %s\n", cudaGetErrorString(cudaStatus));
	}
}
void output(int* M, int N)
{
	for (int i = 0; i < N * N; i++)
	{
		if ((i % N == 0) && (i > 0))
		{
			printf("\n");
		}
		printf("%d ", M[i]);
	}
	printf("\n");
}

int main(void)
{
	cudaError_t cudaStatus;
	cublasStatus_t stat;
	cublasHandle_t handle;
	int i, j, k, N = 2000, choise = 0, error = 0;
	float* a, * b, * c, * test_c, * dev_a, * dev_b, * dev_c;
	printf("input N\n");
	scanf_s("%d", &N);
	clock_t start, end;
	a = (float*)malloc(N * N * sizeof(float));
	b = (float*)malloc(N * N * sizeof(float));
	c = (float*)malloc(N * N * sizeof(float));
	test_c = (float*)malloc(N * N * sizeof(float));
	for (i = 0; i < N * N; i++)
	{
		a[i] = 1; b[i] = 1; c[i] = 0; test_c[i] = 0;
	}
	start = clock();
	printf("Start\n");
	Error(cudaMalloc((void**)&dev_a, N * N * sizeof(float)));
	Error(cudaMalloc((void**)&dev_b, N * N * sizeof(float)));
	Error(cudaMalloc((void**)&dev_c, N * N * sizeof(float)));
	stat = cublasCreate(&handle);
	Error(cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice));
	Error(cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice));
	float al = 1.0f, bet = 0.0f;
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, dev_a, N, dev_b, N, &bet, dev_c, N);
	Error(cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost));
	end = clock();
	printf("End\ntime ON GPU = %f sec\n", ((float)(end - start)) / CLOCKS_PER_SEC);
	printf("check?\n");
	scanf_s("%d", &choise);
	if (choise == 1)
	{
		start = clock();
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				for (k = 0; k < N; k++)
				{
					test_c[i * N + j] += a[i * N + k] * b[k * N + j];
				}
				if (test_c[i * N + j] != c[i * N + j])
				{
					error = 1;
					break;
				}
			}
		}
		end = clock();
		printf("End\ntime ON CPU = %f sec\n", ((float)(end - start)) / CLOCKS_PER_SEC);
	}
	if (choise == 1 && error == 0) printf("check success\n");
	else if (choise == 1 && error != 0) printf("MULT WAS NOT CORRECT!!!\n");
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
	cublasDestroy(handle);
	free(a); free(b); free(c); free(test_c);
	return 0;
}