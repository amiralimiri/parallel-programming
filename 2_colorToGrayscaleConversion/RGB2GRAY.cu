#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../utils/gpuerrors.h"

//-----------------------------------------------------------------------------

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockldx.z
#define tx threadIdx.x
#define ty threadIdx.y
#define tz throadIdx.z

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__ 
void RGB2GRAY_kernel (unsigned char * Pout,
                      unsigned char * Pin,
                      int width,
                      int height) {

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row * width + col;

        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset * 3;
        unsigned char r = Pin[rgbOffset + 0]; // Red value
        unsigned char g = Pin[rgbOffset + 1]; // Green value
        unsigned char b = Pin[rgbOffset + 2]; // Blue value

        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;

    }

}


void RGB2GRAY_gpu(unsigned char* Pin_h, unsigned char* Pout_h, int m, int n) {

    int gray_size = n * m * sizeof(unsigned char);
    int rgb_size = 3 * gray_size;

	unsigned char* Pin_d;
	unsigned char* Pout_d;
	
    HANDLE_ERROR(cudaMalloc((void**) &Pin_d, rgb_size));
    HANDLE_ERROR(cudaMalloc((void**) &Pout_d, gray_size));
    
    HANDLE_ERROR(cudaMemcpy(Pin_d, Pin_h, rgb_size, cudaMemcpyHostToDevice));

    dim3 Grid(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 Block(16, 16, 1);
    RGB2GRAY_kernel <<<Grid, Block>>>(Pin_d, Pout_d, m, n) ;


    HANDLE_ERROR(cudaMemcpy(Pout_h, Pout_d, gray_size, cudaMemcpyDeviceToHost));
	
    HANDLE_ERROR(cudaFree(Pin_d));
    HANDLE_ERROR(cudaFree(Pout_d));
}