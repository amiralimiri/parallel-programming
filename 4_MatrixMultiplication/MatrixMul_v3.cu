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

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__
void MatrixMul_Kernel3 (float* M, float* N, float* P, int width){
    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the P element to work on
    int row = by*TILE_WIDTH + ty;
    int colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < width/TILE_WIDTH; ++ph) {
        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] =M[row*width + ph*TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c*TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            Nds[ty][tx]=N[(ph*TILE_WIDTH + ty) *width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty] [k]*Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c*TILE_WIDTH;
        P[row*width + col] = Pvalue[c];
    }

}









