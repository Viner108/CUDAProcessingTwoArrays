#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"
int main()
{
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* hA;
	float* dA;
	float* hB;
	float* dB;
	float* hC;
	float* dC;

	int nStream = 1; //number of CUDA threads
	int  N_thread = 512;
	int size = N_thread * 50000 / nStream;  //size of each array
	int N_blocks;
	int i;
	unsigned int mem_size = sizeof(float) * size;

	cudaMallocHost((void**)&hA, mem_size * nStream);
	cudaMallocHost((void**)&hB, mem_size * nStream);
	cudaMallocHost((void**)&hC, mem_size * nStream);
	//memory allocation for arrays hA, hB, hC

	cudaMalloc((void**)&dA, mem_size * nStream);
	cudaMalloc((void**)&dB, mem_size * nStream);
	cudaMalloc((void**)&dC, mem_size * nStream);
	//memory allocation in GPU

	for (i = 0; i < size; ++i)
	{
		hA[i] = sinf(i);
		hB[i] = cosf(2.0f*i-5.0f);
		hC[i] = 0.0f;
	}
	//filling arrays

	if ((size % N_thread) == 0) {
		N_blocks = size / N_thread;
	}
	else {
		N_blocks = (int)(size / N_thread) + 1;
	}
	dim3 blocks(N_blocks);
	
	cudaStream_t stream[1];

	for (i = 0; i < nStream; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	//Create CUDA streams

	cudaEventRecord(start, 0);

	for ( i = 0; i < nStream; ++i)
	{
		cudaMemcpyAsync(dA + i * size, hA + i * size, mem_size, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(dB + i * size, hB + i * size, mem_size, cudaMemcpyHostToDevice, stream[i]);
	}
	//asynchronous copying from host to device
 
	for ( i = 0; i < nStream; ++i)
	{
		function << < N_blocks, N_thread, 0, stream[i] >> > (dA + i * size, dB + i * size, dC + i * size, size);
	}

	for (i = 0; i < nStream; ++i)
	{
		cudaMemcpyAsync(hC + i * size, dC + i * size, mem_size, cudaMemcpyDeviceToHost, stream[i]);	
	}
	//asynchronous copying from device to host

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);

	for (i = 0; i < nStream; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
	// destruction of streams

	return 0;
}

