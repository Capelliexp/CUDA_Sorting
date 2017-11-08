#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>

#define DATA_AMOUNT 1000000
#define BLOCK_AMOUNT 666
#define THREADS_PER_BLOCK 666

int main(int argc, char* argv[]) {
	srand(time(NULL));
	int* unsorted, sorted;

	std::cout << "init" << std::endl;

	unsorted = new int[DATA_AMOUNT + 1];
	for (int i = 0; i < DATA_AMOUNT; i++)
		unsorted[i] = (rand() % DATA_AMOUNT) + 1;

	unsorted[DATA_AMOUNT] = DATA_AMOUNT + 1;

	std::cout << "unsorted data created" << std::endl;

	cudaMalloc((void **)&unsorted, DATA_AMOUNT);
	cudaMemcpy(unsorted, dataToSort, DATA_AMOUNT, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&sorted, DATA_AMOUNT);

	std::cout << "transfer to device memory complete" << std::endl;

	//start bubbleSort

	cudaMemcpy(sortedData, sorted, DATA_AMOUNT, cudaMemcpyDeviceToHost);

	std::cout << "transfer to host memory complete" << std::endl;

	cudaFree(unsorted);
	cudaFree(sorted);

	std::cout << "memory freeing complete" << std::endl;

	return 0;
}

__global__
void bubbleSort(int* dataToSort, int* sortedData) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

}