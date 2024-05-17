// Compute vector sum C_h = A_h + B_h
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpuerrors.h"

//-----------------------------------------------------------------------------
void fill(float* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = (float) (rand() % 100);
}

double calc_mse (float* x, float* y, int n) {

	double mse = 0.0;
	int i; for (i=0; i<n; i++) {
		double e = x[i]-y[i];
		e = e * e;
		mse += e;
	}
	mse = mse / n;
	return mse;
}

//-----------------------------------------------------------------------------
void vecAdd_cpu(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

//-----------------------------------------------------------------------------
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockldx.z
#define tx threadIdx.x
#define ty threadIdx.y
#define tz throadIdx.z

// Compute vector sum C = A + B 
// Each thread performs one pair-wise addition 
__global__ 
void vecAdd_Kernel(float* A, float* B, float* C, int n) { 
    int i =  bx * blockDim.x + tx; 
    if (i < n) { 
        C[i] = A[i] + B[i]; 
    } 
}

void vecAdd_gpu(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);

	float* A_d;
	float* B_d;
	float* C_d;
	
    HANDLE_ERROR(cudaMalloc((void**) &A_d, size));
    HANDLE_ERROR(cudaMalloc((void**) &B_d, size));
    HANDLE_ERROR(cudaMalloc((void**) &C_d, size));
    
    HANDLE_ERROR(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

	dim3 Grid(ceil(n/256.0), 1, 1);
	dim3 Block(256, 1, 1);
    vecAdd_Kernel <<<Grid, Block>>>(A_d, B_d, C_d, n);

    HANDLE_ERROR(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
	
    HANDLE_ERROR(cudaFree(A_d));
    HANDLE_ERROR(cudaFree(B_d));
    HANDLE_ERROR(cudaFree(C_d));
}

//-----------------------------------------------------------------------------
int steps[10]={16,32,64,128,256,512,1024,2048,4096,8192};

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
    
	for(int i=0; i<sizeof(steps)/sizeof(int) ; i++) {

		float* a;
		float* b;
		float* c;
		float* c_serial;

		int n = steps[i];
		
		a        = (float*)malloc(n * sizeof(float));
		b        = (float*)malloc(n * sizeof(float));
		c        = (float*)malloc(n * sizeof(float));
		c_serial = (float*)malloc(n * sizeof(float));
						
		srand(0);
		fill(a, n);
		fill(b, n);

		clock_t t0 = clock(); 
		vecAdd_cpu(a, b, c_serial, n);
		clock_t t1 = clock(); 
		
		clock_t t2 = clock(); 
		vecAdd_gpu(a, b, c, n);
		clock_t t3 = clock();
		
		float mse = calc_mse(c_serial, c, n);

		printf("n=%d\t CPU=%06lu ms GPU=%06lu ms mse=%f\n",n, (t1-t0)/1000, (t3-t2)/1000, mse);
		
		free(a);
		free(b);
		free(c);
		free(c_serial);
	}
    
	cudaThreadExit();
	getchar();
	return 0;
}

