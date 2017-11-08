#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>

#define DATA_AMOUNT 1000000
#define BLOCK_AMOUNT 24
#define THREADS_PER_BLOCK 1

const dim3 blockSize = dim3(BLOCK_AMOUNT, 1, 1);
const dim3 threadsPerBlock = dim3(THREADS_PER_BLOCK, 1, 1);

__global__
void bubbleSort(int* data_d) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (data_d[i * 2] > data_d[i * 2 + 1]) {
		int temp = data_d[i * 2];
		data_d[i * 2] = data_d[i * 2 + 1];
		data_d[i * 2 + 1] = temp;
	}
	__syncthreads();
	if (data_d[i * 2 + 1] > data_d[i * 2 + 2]) {
		int temp = data_d[i * 2 + 1];
		data_d[i * 2 + 1] = data_d[i * 2 + 2];
		data_d[i * 2 + 2] = temp;
	}
}

int main(int argc, char* argv[]) {
	srand(time(NULL));
	int* data = new int[DATA_AMOUNT + 1];
	int* data_d;
	

	std::cout << "init" << std::endl;

	for (int i = 0; i < DATA_AMOUNT; i++)
		data[i] = (rand() % DATA_AMOUNT) + 1;

	data[DATA_AMOUNT] = DATA_AMOUNT + 1;
	std::cout << "unsorted data created" << std::endl;

	cudaMalloc((void**)&data_d, DATA_AMOUNT * sizeof(int));
	cudaMemcpy(data_d, data, DATA_AMOUNT * sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "transfer to device memory complete" << std::endl;

	std::cout << "sort start" << std::endl;
	bubbleSort <<<blockSize, threadsPerBlock >>>(data_d);
	std::cout << "sort fin" << std::endl;

	cudaMemcpy(&data, &data_d, DATA_AMOUNT, cudaMemcpyDeviceToHost);
	std::cout << "transfer memory from device to host complete" << std::endl;
	
	cudaFree(data_d);
	std::cout << "memory freeing complete" << std::endl;
	
	std::cout << "Res: ";
	for (int i = 0; (i < DATA_AMOUNT) && (i < 50); i++)
		std::cout << data[i] << ", ";
	std::cout << std::endl;

	getchar();
	return 0;
}