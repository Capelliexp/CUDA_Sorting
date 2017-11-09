#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>

#define DATA_AMOUNT 10
#define BLOCK_AMOUNT 16
#define THREADS_PER_BLOCK 5

const dim3 blockSize = dim3(BLOCK_AMOUNT, 1, 1);
const dim3 threadsPerBlock = dim3(THREADS_PER_BLOCK, 1, 1);

__global__
void bubbleSort(int* data_d, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int iter = i * 2;

	if (i < n) {
		if (data_d[iter] > data_d[iter + 1]) {
			int temp = data_d[iter];
			data_d[iter] = data_d[iter + 1];
			data_d[iter + 1] = temp;
		}
		iter++;
		__syncthreads();
		if (data_d[iter] > data_d[iter + 1]) {
			int temp = data_d[iter];
			data_d[iter] = data_d[iter + 1];
			data_d[iter + 1] = temp;
		}
	}
}

int main(int argc, char* argv[]) {
	srand(time(NULL));
	int* data = new int[DATA_AMOUNT + 1];
	int* data_d = 0;
	//int* dataSorted;	//not required
	int dataSorted[DATA_AMOUNT] = {0}; //OBS! potential stack overflow
	
	//---

	std::cout << "init" << std::endl;

	for (int i = 0; i < DATA_AMOUNT; i++)
		data[i] = (rand() % DATA_AMOUNT) + 1;

	data[DATA_AMOUNT] = (int)DATA_AMOUNT + 1;	//depends on even/odd of DATA_AMOUNT
	std::cout << "unsorted data created" << std::endl;

	std::cout << "Pre sort: ";
	for (int i = 0; (i < DATA_AMOUNT) && (i < 50); i++)
		std::cout << data[i] << ", ";
	std::cout << data[DATA_AMOUNT] << std::endl;

	//---

	cudaMalloc((void**)&data_d, DATA_AMOUNT * sizeof(int));
	cudaMemcpy(data_d, data, DATA_AMOUNT * sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "transfer to device memory complete" << std::endl;

	std::cout << "sort start" << std::endl;
	bubbleSort <<<blockSize, threadsPerBlock >>>(data_d, (int)DATA_AMOUNT);
	std::cout << "sort fin" << std::endl;

	cudaMemcpy(&dataSorted, data_d, DATA_AMOUNT * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "transfer memory from device to host complete" << std::endl;
	
	cudaFree(data_d);
	std::cout << "memory freeing complete" << std::endl;
	
	//---

	std::cout << "Post sort: ";
	for (int i = 0; (i < DATA_AMOUNT) && (i < 50); i++)
		std::cout << dataSorted[i] << ", ";
	std::cout << data[DATA_AMOUNT] << std::endl;

	getchar();
	return 0;
}