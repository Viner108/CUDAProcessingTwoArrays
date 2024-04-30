#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void function(float* dA, float* dB, float* dC, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;
	float ab, sum = 0.f;
	if (i < size) {
		ab = dA[i] * dB[i];
		for ( j = 0; j < 100; j++)
		{
			sum = sum + sinf(sinf(j + ab));
			dC[i] = sum;
		}
	}
}