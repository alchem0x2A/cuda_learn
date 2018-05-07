#ifndef _MY_CUDA_KERNEL_
#define _MY_CUDA_KERNEL_
#include "my.h"
__global__ void mykernel(char *a, const char *b, const int length){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length){
    a[i] = b[i];
  }
};

__global__ void my_N_kernel(char *a, const char *b,
			    const int length, const int N){
  // copy the content of b to a N times
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int start, offset;
  if (i < N){
    start = (length - 1) * i;
    for (offset = 0; offset < (length - 1); ++offset)
      a[start + offset] = b[offset];
  }
};
#endif
