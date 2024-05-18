#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("number of available CUDA devices: %d\n", devCount);

    struct cudaDeviceProp p;
    for(unsigned int i = 0; i < devCount; i++) {
        printf("###############################################\n");
        cudaGetDeviceProperties(&p, i);
        printf("Device Name: %s\n", p.name);
        printf(" > Max Threads Per Block: %d\n", p.maxThreadsPerBlock);
        printf(" > number of SMs: %d\n", p.multiProcessorCount);
        printf(" > clock frequency of the device: %d\n", p.clockRate);
        printf(" > maximum number of threads allowed along each dimension of a block: %d %d %d\n",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf(" > maximum number of blocks allowed along each dimension of a grid: %d %d %d\n",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf(" > number of registers that are available in each SM: %d\n", p.regsPerBlock);
        printf(" > size of warps: %d\n", p.warpSize);
    }
    cudaThreadExit();
	getchar();
	return 0;
}