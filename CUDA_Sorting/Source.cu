#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <windows.h>
#include <time.h>
#include <ctime>

void PrintGenInfo();
void FillArray(int data[]);
int* SortCUDA(int* data);
int* SortCPU(int* data);
void SortElements(int* data, int pos);
void CheckDataOrder(int dataSorted[]);
void PrintError(int zone, cudaError_t* blob);

#define DATA_AMOUNT 50000
#define BLOCK_AMOUNT 1024		//max 65535 ???
#define THREADS_PER_BLOCK 128	//max 1024 ???

const dim3 blockSize = dim3(BLOCK_AMOUNT, 1, 1);
const dim3 threadsPerBlock = dim3(THREADS_PER_BLOCK, 1, 1);

//DEVICE
__global__
void OddEvenSortv1(int* data_d, int n) {
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

__global__
void OddEvenSortv2(int* data_d, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	long iter = i * 2;
	int evenSwitch = 0;

	int compLeft = data_d[iter];
	int compMid = data_d[iter + 1];

	if (iter < n) {
		if (compLeft > compMid) {
			int temp = compLeft;
			compLeft = compMid;
			compMid = temp;
			data_d[iter] = (int)compLeft;
			evenSwitch++;
		}
		
		iter++;
		__syncthreads();

		int compRight = data_d[iter + 1];

		if (compMid > compRight) {
			data_d[iter] = compRight;
			data_d[iter + 1] = compMid;
		}
		else if(evenSwitch > 0) {
			data_d[iter] = compMid;
		}
	}
}

//HOST
int main(int argc, char* argv[]) {
	int* data = new int[DATA_AMOUNT + 1];
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

	std::cout << "DATA_AMOUNT: " << DATA_AMOUNT << std::endl <<
		"BLOCK_AMOUNT: " << BLOCK_AMOUNT << std::endl <<
		"THREADS_PER_BLOCK: " << THREADS_PER_BLOCK << std::endl;

	if ((BLOCK_AMOUNT*THREADS_PER_BLOCK * 2) < DATA_AMOUNT)
		std::cout << std::endl << "!!! WARNING: not enough threads to cover data length !!!" << std::endl;

	std::cout << std::endl;
}

//FILL DATA WITH RAND NUMBERS
void FillArray(int data[]) {
	for (int i = 0; i < DATA_AMOUNT; i++)
		data[i] = (rand() % DATA_AMOUNT) + 1;

	if (!(DATA_AMOUNT % 2))
		data[DATA_AMOUNT] = (int)DATA_AMOUNT + 1;

	
	/*std::cout << "Pre sort:   ";
	for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 7); i++)
		std::cout << data[i] << ", ";
	std::cout << data[DATA_AMOUNT - 1] << (DATA_AMOUNT > 10 ? "..." : "") << std::endl;*/
}

//SORT DATA WITH CUDA USING ODD-EVEN SORTING
int* SortCUDA(int* data) {
	srand(time(NULL));
	std::clock_t start;
	double duration;
	int* dataSorted = new int[DATA_AMOUNT];
	int* data_d = 0;
	cudaError_t blob;


	if ((blob = cudaMalloc((void**)&data_d, (DATA_AMOUNT + 1) * sizeof(int))) != cudaSuccess) PrintError(0, &blob);
	if ((blob = cudaMemcpy(data_d, data, (DATA_AMOUNT + 1) * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) PrintError(1, &blob);

	start = std::clock();	//timer start
	for (int i = 0; i < (DATA_AMOUNT / 2) + 1; i++)
		OddEvenSortv2 <<<blockSize, threadsPerBlock >> > (data_d, (int)DATA_AMOUNT);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;	//timer stop
	std::cout << "CUDA sorting time: " << duration << " sec" << std::endl;

	if ((blob = cudaMemcpy(dataSorted, data_d, (DATA_AMOUNT) * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) PrintError(2, &blob);
	if ((blob = cudaFree(data_d)) != cudaSuccess) PrintError(3, &blob);

	return dataSorted;
}

//SORT DATA WITH CPU USING ODD-EVEN SORTING
int* SortCPU(int* data) {
	srand(time(NULL));
	std::clock_t start;
	double duration;
	int* dataSorted = data;

	start = std::clock();	//timer start
	float checkpoint = 10;	//%

	for (int i = 0; i < DATA_AMOUNT; i++) {
		float rand = ((i*100) / DATA_AMOUNT);
		if (rand > checkpoint) {
			std::cout << checkpoint << "%" << std::endl;
			checkpoint += 10;
		}
		for (int j = 0; j < DATA_AMOUNT - i; j++)
			SortElements(dataSorted, j);
	}

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;	//timer stop
	std::cout << "CPU sorting time: " << duration << " sec" << std::endl;

	return dataSorted;
}

//SORTING ALGORITHM USED BY CPU
void SortElements(int* data, int pos) {
	if (data[pos] > data[pos+1]) {
		int temp = data[pos];
		data[pos] = data[pos + 1];
		data[pos + 1] = temp;
	}
}

//CHECK RESULT
void CheckDataOrder(int dataSorted[]) {
	/*std::cout << "Post sort:  ";
	for (int i = 0; (i < DATA_AMOUNT - 1) && (i < 7); i++)
		std::cout << dataSorted[i] << ", ";
	std::cout << dataSorted[DATA_AMOUNT - 1] << (DATA_AMOUNT > 10 ? "..." : "") << std::endl;*/
	
	int unsortedCount = 0;
	for (int i = 0; i < DATA_AMOUNT - 1; i++)
		if (dataSorted[i] > dataSorted[i + 1]) {
			unsortedCount++;
		}

	if (unsortedCount > 0)
		std::cout << "WARNING: " << unsortedCount << " elements unsorted" << std::endl;

	if (dataSorted[0] < 0 || dataSorted[DATA_AMOUNT-1] < 0)
		std::cout << "WARNING: incorrect array values" << std::endl;
	
	if(!(unsortedCount > 0) && !(dataSorted[0] < 0 || dataSorted[DATA_AMOUNT - 1] < 0))
		std::cout << "list sorted correctly" << std::endl;
}

//CUDA ERROR TYPE PRINT 
void PrintError(int zone, cudaError_t* blob) {
	std::cout << "ERROR: ";
	switch (zone) {
	case 0:
		std::cout << "MALLOC 1 - ";
		break;
	case 1:
		std::cout << "MEMCPY 1 - ";
		break;
	case 2:
		std::cout << "MEMCPY 2 - ";
		break;
	case 3:
		std::cout << "FREE 1 - ";
		break;
	default:
		std::cout << "<unhandled zone> - ";
		break;
	}
	std::cout << cudaGetErrorString(*blob) << std::endl;
	*blob = cudaDeviceReset();
}