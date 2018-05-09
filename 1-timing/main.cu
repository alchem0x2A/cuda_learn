#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>

__global__ void simple_op(float* a, const long length){
  long total_threads = gridDim.x * blockDim.x;
  long chunksize = length / total_threads > 0 ? length / total_threads : 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  int start = i * chunksize;
  if (start < length){
    int end = (i + 1) * chunksize > length ? length : (i + 1) * chunksize;
    for (j = start; j < end; ++j)
      a[j] = a[j] * 2;
  }
}

float serial(float *a, long length){
  clock_t start = clock();
  long i;
  for(i = 0; i < length; ++i)
    a[i] = a[i] * 2;
  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;
  return seconds;
}

float CUDA(float *a, long length, int grid, int block){
  clock_t start = clock();
  float *dev_a;
  cudaMalloc((void **) &dev_a, length * sizeof(float));
  cudaMemcpy(dev_a, a, length * sizeof(float), cudaMemcpyHostToDevice);
  simple_op<<<grid, block>>>(dev_a, length);
  cudaMemcpy(a, dev_a, length * sizeof(float), cudaMemcpyDeviceToHost);
  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;
  cudaFree(dev_a);
  return seconds;
}

int main(int argc, char* argv[]){
  if (argc < 2){
    std::cerr << "timing N" << std::endl;
    return 1;
  }

  int size_2_ = std::stoi(argv[1]);
  long length = 1 << size_2_;	// 2 ^ size_2_
  float *a = new float[length];

  // Serial code
  std::fill(a, &a[length], 1.0);
  float t1 = 0;
  for (int i = 0; i < 5; ++i){
    t1 += serial(a, length);
  }
  t1 = t1 / 5;

  // CUDA code

  std::fill(a, &a[length], 1.0);
  float t2 = 0;
  int grid = 16;
  int block = 256;
  for (int i = 0; i < 5; ++i){
    t2 += CUDA(a, length, grid, block);
  }
  t2 = t2 / 5;

  std::cout << t1 << " " << t2 << std::endl;
  return 0;
  
}