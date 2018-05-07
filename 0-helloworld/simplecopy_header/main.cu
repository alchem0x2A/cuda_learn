#include <iostream>
#include "my.h"
#define SIZE 14

int main(int argc, char* argv[]){
  int size = SIZE;
  int N = 5;
  int long_size = (size - 1) * N + 1;
  char origin[SIZE] = "Hello World!\n";
  char *copy = (char *) malloc(size);
  char *long_copy = (char *) malloc(long_size);
  memset(copy, '\0', size);
  memset(long_copy, '\0', size);
  std::cout << "Original Value of Copy:" << std::endl;
  std::cout << copy << std::endl;
  
  char *dev_origin, *dev_copy, *dev_longcopy;
  // Malloc & Copy data
  cudaMalloc((void **)&dev_origin, size * sizeof(char));
  cudaMalloc((void **)&dev_copy, size * sizeof(char));
  cudaMalloc((void **)&dev_longcopy, long_size * sizeof(char));
  cudaMemcpy(dev_origin, origin, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_copy, copy, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_longcopy, long_copy, long_size, cudaMemcpyHostToDevice);
  
  // execute code
  // Must use device copy of the chars!!
  mykernel<<<10, 10>>>(dev_copy, dev_origin, size);
  my_N_kernel<<<10, 10>>>(dev_longcopy, dev_origin, size, N);

  // copy back
  cudaMemcpy(copy, dev_copy, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(long_copy, dev_longcopy, long_size, cudaMemcpyDeviceToHost);
  
  std::cout << "Copied Value of Copy:" << std::endl;
  std::cout << copy;
  
  std::cout << "Copied Value of " << N 
	    << " Copies:" << std::endl;
  std::cout << long_copy;
  
  free(copy);
  cudaFree(dev_origin);
  cudaFree(dev_copy);
  
  return 0;
}
