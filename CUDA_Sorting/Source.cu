#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <windows.h>
//#include <time.h>
#include <ctime>

void PrintGenInfo();
void FillArray(int data[]);
int* SortCUDA(int* data);
int* SortCPU(int* data);
void SortElements(int* data, int pos, int mod);
void CheckDataOrder(int dataSorted[]);
void PrintError(std::string zone, cudaError_t* blob);

//https://stackoverflow.com/questions/5856250/using-nsight-to-debug

#define DATA_AMOUNT 7500
#define BLOCK_AMOUNT 1
#define THREADS_PER_BLOCK 1000 //max 1024
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)	//max 131072 ???

const dim3 blockSize = dim3(BLOCK_AMOUNT, 1, 1);
const dim3 threadsPerBlock = dim3(THREADS_PER_BLOCK, 1, 1);

//DEVICE
__global__
void OddEvenSort(int* data_d, int iterAmount) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if ((id *= iterAmount) < DATA_AMOUNT) {
		int mod = 0;
		
		for (int i = 0; i < DATA_AMOUNT; ++i) {
			__syncthreads();
			for (int j = 0; j < iterAmount; j += 2) {
				int pos = id + mod + j;
				if ((pos + 1) < DATA_AMOUNT) {
					int reg1 = data_d[pos];
					int reg2 = data_d[pos + 1];
					if (reg1 > reg2) {
						data_d[pos] = reg2;
						data_d[pos + 1] = reg1;
					}
					/*if (data_d[pos] > data_d[pos + 1]) {
						int temp = data_d[pos];
						data_d[pos] = data_d[pos + 1];
						data_d[pos+1] = temp;
					}*/
				}
				
			}
			if (mod == 0) mod = 1;
			else mod = 0;
		}
		
	}
	
}

__global__
void Nothing(int* data_d, int iterAmount) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	id *= iterAmount;

	for (int i = 0; i < iterAmount; i++)
		data_d[id + i] = data_d[id + i + 1] + id + i;
}

//HOST
int main(int argc, char* argv[]) {
	int* data = new int[DATA_AMOUNT];
	int* dataSortedByCUDA = new int[DATA_AMOUNT];
	int* dataSortedByCPU = new int[DATA_AMOUNT];

	PrintGenInfo();
	FillArray(data);

	dataSortedByCUDA = SortCUDA(data);
	CheckDataOrder(dataSortedByCUDA);

	std::cout << std::endl;
	dataSortedByCPU = SortCPU(data);	
	CheckDataOrder(dataSortedByCPU);

	std::cout << std::endl << "fin" << std::endl;
	getchar();
	return 0;
}

//PRINT INFORMATION AND WARNINGS
void PrintGenInfo() {
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo);
	cudaError_t blob;

	int deviceCount = 0, setDevice = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) PrintError("Get1", &blob);
	if (cudaSetDevice(setDevice) != cudaSuccess) PrintError("Set1", &blob);

	std::cout <<
		"DATA_AMOUNT: " << DATA_AMOUNT << std::endl <<
		"BLOCK_AMOUNT: " << BLOCK_AMOUNT << std::endl <<
		"THREADS_PER_BLOCK: " << THREADS_PER_BLOCK << std::endl <<
		std::endl << "Available devices: " << deviceCount << std::endl <<
		"Device ID in use: <" << setDevice << ">" << std::endl;


	if ((THREADS_PER_BLOCK * 2) < DATA_AMOUNT) {
		int iterAmount = 2;
		while (1) {
			if (THREADS_PER_BLOCK*iterAmount >= DATA_AMOUNT)
				break;
			iterAmount += 2;
		}
		std::cout << std::endl << "WARNING: low thread count forces " << iterAmount/2 << " thread loops" << std::endl;
	}
		
	std::cout << std::endl;
}

//FILL DATA WITH RAND NUMBERS
void FillArray(int data[]) {
	srand(time(NULL));
	for (int i = 0; i < DATA_AMOUNT; i++)
		data[i] = (rand() % DATA_AMOUNT) + 1;

	/*
	std::cout << "Pre sort:   ";
	for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 10); i++)
		std::cout << data[i] << ", ";
	std::cout << data[10] << (DATA_AMOUNT > 10 ? "..." : "") << std::endl << std::endl;
	*/
}

//SORT DATA WITH CUDA USING ODD-EVEN SORTING
int* SortCUDA(int* data) {
	int* dataSorted = new int[DATA_AMOUNT];
	//int* dataSorted = (int*)malloc(DATA_AMOUNT * sizeof(int));
	int* data_d = 0;
	cudaError_t blob;
	int iterAmount = 2;
	cudaEvent_t start, stop;
	float duration = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	while (1) {
		if (THREADS_PER_BLOCK*iterAmount >= DATA_AMOUNT)
			break;
		iterAmount += 2;
	}

	if ((blob = cudaMalloc((void**)&data_d, DATA_AMOUNT * sizeof(int))) != cudaSuccess) PrintError("Mal1", &blob);
	if ((blob = cudaMemcpy(data_d, data, DATA_AMOUNT * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) PrintError("Cpy1", &blob);

	cudaEventRecord(start);
	OddEvenSort<<<BLOCK_AMOUNT, THREADS_PER_BLOCK>>>(data_d, iterAmount);
	//Nothing<<<BLOCK_AMOUNT, THREADS_PER_BLOCK>>>(data_d, iterAmount);
	cudaEventRecord(stop);

	if ((blob = cudaMemcpy(dataSorted, data_d, DATA_AMOUNT * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) PrintError("Cpy2", &blob);
	if ((blob = cudaFree(data_d)) != cudaSuccess) PrintError("Free1", &blob);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);

	std::cout << "CUDA sorting time: " << duration/1000 << " sec" << std::endl;
	return dataSorted;
}

//SORT DATA WITH CPU USING ODD-EVEN SORTING
int* SortCPU(int* data) {
	std::clock_t start;
	int* dataSorted = data;
	int mod;

	start = std::clock();	//timer start
	for (int i = 0; i < DATA_AMOUNT; ++i) {	//will run DATA_AMOUNT times
		mod = i%2;
		for (int j = 0; j < DATA_AMOUNT-1; j +=2)	//will run DATA_AMOUNT/2 times
			SortElements(dataSorted, j, mod);
	}
	long double duration = (std::clock() - start) / (long double)CLOCKS_PER_SEC;	//timer stop

	std::cout << "CPU sorting time: " << duration << " sec" << std::endl;
	return dataSorted;
}

//SORTING ALGORITHM USED BY CPU
void SortElements(int* data, int pos, int mod) {
	pos += mod;
	if (data[pos] > data[pos+1] && pos + 1 < DATA_AMOUNT) {
		int temp = data[pos];
		data[pos] = data[pos + 1];
		data[pos + 1] = temp;
	}
}

//CHECK RESULT
void CheckDataOrder(int dataSorted[]) {
	/*std::cout << "Post sort:  ";
	for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 10); i++)
		std::cout << dataSorted[i] << ", ";
	std::cout << dataSorted[10] << (DATA_AMOUNT > 10 ? "..." : "") << std::endl;*/

	int unsortedCount = 0;
	bool incorrectValues = false;
	for (int i = 0; i < DATA_AMOUNT-1; i++) {
		if (dataSorted[i] > dataSorted[i + 1]) {
			unsortedCount++;
			//std::cout << "it:" << i << " - ... " << dataSorted[i - 2] << ", " << dataSorted[i - 1] << ", " << dataSorted[i] << ", " << dataSorted[i + 1] << ", " << dataSorted[i + 2] << " ..." << std::endl;
		}
		if (dataSorted[i] > DATA_AMOUNT || dataSorted[i] < 0)
			incorrectValues = true;
	}

	if (unsortedCount > 0)
		std::cout << std::endl << "WARNING: " << unsortedCount << " elements unsorted" << std::endl;

	if (incorrectValues == true)
		std::cout << "WARNING: incorrect array values" << std::endl;
	
	if(!(unsortedCount > 0) && !(dataSorted[0] < 0 || dataSorted[DATA_AMOUNT - 1] < 0))
		std::cout << "list sorted correctly" << std::endl;
}

//CUDA ERROR TYPE PRINT 
void PrintError(std::string zone, cudaError_t* blob) {
	std::cout << "ERROR" << std::endl << "   zone: " << zone << std::endl << "   output: " << cudaGetErrorString(*blob) << std::endl;
	*blob = cudaDeviceReset();
}