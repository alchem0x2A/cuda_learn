#include <iostream>
#include <algorithm>




// Swap by reference of pointers, only c++
template<typename T>
void cudaSwap1(T *&a, T *&b){
  T *tmp = a;
  a = b;
  b = tmp;
}

template<typename T>
void cudaSwap2(T **a, T **b){
  // Swap the memory if pointers a and b
  T *tmp = *a;
  *a = *b;
  *b = tmp;
}

 // Won't work on device only
template<typename T>
__global__ void cudaSwap_kernel(T **a, T **b){
  T *tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
  
}

// Very long manipulation on GPU side
template<typename T>
__global__ void manipulate(T *a, T *b, int len){
  
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid < len){
    T a_local = a[pid];
    // presumably a long kernel
    for (int i=0; i < len; ++i){
      // for (int j=0; j < 10000; ++j)
	a_local = a[pid] + 2.0 * b[i];      
    }
    a[pid] = a_local;
  }
}



int main(int argc, char* argv[]){
  int N = 256 * 256;
  int N_th = 1024;
  float *a = new float[N];
  float *b = new float[N];
  std::fill(a, &a[N], 0.0f);
  std::fill(b, &b[N], 1.0f);

  float *da, *db;
  cudaMalloc((void **) &da, N * sizeof(float));
  cudaMalloc((void **) &db, N * sizeof(float));

  std::cout << a[0] << " " << b[0] << std::endl;
  cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int i=0; i< 5; ++i){
    manipulate<<<(N + N_th - 1) / N_th, N_th>>>(da, db, N);
    // All host-side code works
    // Seems no need to use cudaSyncthreads()

    // cudaSwap2(&da, &db);
    // std::swap(da, db);
  
    // This won't work
    // cudaSwap_kernel<<<1, 1>>>(&da, &db);
    cudaSwap1(da, db);
  }




  cudaMemcpy(a, da, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, db, N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << a[0] << " " << b[0] << std::endl; // expect to output 29 70
  delete a; delete b;
  cudaFree(da); cudaFree(db);
  
  return 0;
}
