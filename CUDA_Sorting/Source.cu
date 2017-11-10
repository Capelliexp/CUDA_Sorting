#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>
#include <ctime>

#define DATA_AMOUNT 180
#define BLOCK_AMOUNT 16
#define THREADS_PER_BLOCK 5

const dim3 blockSize = dim3(BLOCK_AMOUNT, 1, 1);
const dim3 threadsPerBlock = dim3(THREADS_PER_BLOCK, 1, 1);

__global__
void bubbleSort(int* data_d, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int iter = i * 2;

	if (iter < n) {
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
	std::clock_t start;
	double duration;
	int* data = new int[DATA_AMOUNT + 1];
	int* dataSorted = new int[DATA_AMOUNT];
	int* data_d = 0;
	
	std::cout << "init" << std::endl;

	//FILL DATA WITH RAND NUMBERS

	for (int i = 0; i < DATA_AMOUNT; i++)
		data[i] = (rand() % DATA_AMOUNT) + 1;
	
	if(!(DATA_AMOUNT%2))
		data[DATA_AMOUNT] = (int)DATA_AMOUNT + 1;

	if (DATA_AMOUNT < 15) {
		std::cout << "Pre sort:   ";
		for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 50); i++)
			std::cout << data[i] << ", ";
		std::cout << data[DATA_AMOUNT - 1] << std::endl;
	}

	//ALLOCATE SPACE ON DEVICE, EXECUTE SORTING AND IMPORT RESULT

	cudaMalloc((void**)&data_d, (DATA_AMOUNT + 1) * sizeof(int));
	cudaMemcpy(data_d, data, (DATA_AMOUNT + 1) * sizeof(int), cudaMemcpyHostToDevice);

	start = std::clock();	//timer start
	for(int i = 0; i < (DATA_AMOUNT/2); i++)
		bubbleSort <<<blockSize, threadsPerBlock >>>(data_d, (int)DATA_AMOUNT);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;	//timer stop

	cudaMemcpy(dataSorted, data_d, (DATA_AMOUNT) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(data_d);
	
	//CHECK RESULT AND END

	if (DATA_AMOUNT < 15) {
		std::cout << "Post sort:  ";
		for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 50); i++)
			std::cout << dataSorted[i] << ", ";
		std::cout << dataSorted[DATA_AMOUNT - 1] << std::endl;
	}
	else {
		int unsortedCount = 0;
		for (int i = 0; i < DATA_AMOUNT-1; i++) 
			if (dataSorted[i] > dataSorted[i + 1]) {
				std::cout << "   " << i << ": " << dataSorted[i] << " > " << dataSorted[i + 1] << std::endl;
				unsortedCount++;
			}
				
		if (unsortedCount > 0)
			std::cout << "WARNING: " << unsortedCount << " elements unsorted" << std::endl;
	}

	std::cout << "sorting time: " << duration << '\n';
	std::cout << "fin";

	getchar();
	return 0;
}