#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../utils/gpuerrors.h"

int BLUR_SIZE = 1; // 3*3
//-----------------------------------------------------------------------------

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockldx.z
#define tx threadIdx.x
#define ty threadIdx.y
#define tz throadIdx.z

__global__
void blur_Kernel (unsigned char *in,
                  unsigned char *out,
                  int width,
                  int height) {

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        // Get average of the surrounding BLUR SIZE X BLUR SIZE box
        for (int blurRow =- BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow) {
            for (int blurCol =- BLUR_SIZE; blurCol<BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Verify we have a valid image pixel
                if(curRow>=0 && curRow<height && curCol>=0 && curCol<width) {
                    pixVal += in[curRow*width + curCol];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }
        }
        // Write our new pixel value out
        out[row*width + col] =(unsigned char) (pixVal/pixels);

    }
}