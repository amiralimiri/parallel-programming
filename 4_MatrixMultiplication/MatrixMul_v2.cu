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

#define TILE_WIDTH 16

__global__
void MatrixMul_Kernel2 (float* M, float* N, float* P, int Width) {

    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH +ty) *Width + Col];
        __syncthreads();

        for (int k=0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;
}