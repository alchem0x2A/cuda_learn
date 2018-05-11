#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include <cassert>

// Reduce from *data (size len) to *result
template<typename T>
__global__ void cudaReduceThread(const T *data, T *result, int len){
  extern __shared__ T shared_data[];	//
  int pid = blockIdx.x * blockDim.x + threadIdx.x; // global thread
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Copy global data to shared mem

  if (pid < len){
    shared_data[tid] = data[pid];
  }
  else{
    shared_data[tid] = 0.0f;	// make sure no overlap
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2){
    // if (tid % (2 * s) == 0){
      // shared_data[tid] += shared_data[tid + s];
    // }
    int ind = tid * 2 * s;	// avoid fork
    if (ind < blockDim.x){
      shared_data[ind] += shared_data[ind + s];
    }
    __syncthreads();
  }
  if (tid == 0){
    result[bid] = shared_data[0];
  }
}

// Summation by 2 rounds of CUDA block reduction
template<typename T>
T cudaReduction(T *data, int len){
  int N_thread = 1024;
  int N_block = (len + N_thread - 1) / N_thread;
  int N_thread2 = 1;
  while (N_thread2 < N_block) N_thread2 *= 2;
  std::cout << "Number of blocks: " << N_block << std::endl;
  float *ddata; cudaMalloc((void **)&ddata, len * sizeof(float));

  cudaMemcpy(ddata, data, len * sizeof(float), cudaMemcpyHostToDevice);
  
  T *block_sum = new T[N_block];
  std::fill(block_sum, &block_sum[N_block], 0.0f);
  T *dblock_sum; cudaMalloc((void **)&dblock_sum, N_block * sizeof(T));

  cudaMemcpy(dblock_sum, block_sum, N_block * sizeof(T),
	     cudaMemcpyHostToDevice);

  // First run on thread
  cudaReduceThread<<<N_block, N_thread, N_thread * sizeof(T)>>>(ddata, dblock_sum, len);
  // Second run on block
  cudaReduceThread<<<1, N_thread2, N_thread2 * sizeof(T)>>>(dblock_sum, dblock_sum, N_block);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      std::cerr << "Failed to use kernel: "
		<< cudaGetErrorString(err)
		<< std::endl;
      exit(EXIT_FAILURE);
    }

  cudaMemcpy(block_sum, dblock_sum, N_block * sizeof(T),
	     cudaMemcpyDeviceToHost);
  T host_sum = block_sum[0];
  // std::cout << "Sum each block:" << std::endl;
  // for (int i = 0; i < N_block; ++i){
    // std::cout << block_sum[i] << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < N_block; ++i)
    // host_sum += block_sum[i];
  delete block_sum;
  cudaFree(dblock_sum); cudaFree(ddata);
  return host_sum;
}

int main(int argc, char* argv[]){
  const int N = 1e5;
  float *data = new float[N];
  float sum_serial = 0.0f;
  // Naive addition serial
  for (int i=0; i < N; ++i){
    float f_ = 0.1f;
    data[i] = f_;
    sum_serial += f_;
  }

  std::cout << "sum by serial: " << sum_serial << std::endl;
  
  float sum = cudaReduction(data, N);
  std::cout << "sum by CUDA: " << sum << std::endl;

  assert((sum - sum_serial) / sum_serial < 1e-5); // Round-off error possible
  delete data;
  
  return 0;
}
