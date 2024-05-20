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

__global__
void MatrixMul_Kernel1 (float* M, float* N, float* P, int Width) {

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if ((row < Width) && (col < Width) ) {
        float Pvalue = 0;
        for(int k = 0; k < Width; ++k){
            Pvalue += M[row*Width+k] * N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}