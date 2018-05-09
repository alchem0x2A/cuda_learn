#include <iostream>
#include <algorithm>


// The swap of pointer should not be done at the device side,
// rather the host side!
// Only pass by reference works
template<typename T>
void cudaSwap(T *&a, T *&b){
  // Swap the memory if pointers a and b
  T *tmp = a;
  a = b;
  b = tmp;
}



int main(int argc, char* argv[]){
  float *a = new float[10];
  float *b = new float[10];
  std::fill(a, &a[10], 1.0f);
  std::fill(b, &b[10], 2.0f);

  float *da, *db;
  cudaMalloc((void **) &da, 10 * sizeof(float));
  cudaMalloc((void **) &db, 10 * sizeof(float));

  std::cout << a[0] << " " << b[0] << std::endl;
  cudaMemcpy(da, a, 10 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, 10 * sizeof(float), cudaMemcpyHostToDevice);

  // both works
  // cudaSwap(da, db);
  std::swap(da, db);

  cudaMemcpy(a, da, 10 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, db, 10 * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << a[0] << " " << b[0] << std::endl;
  delete a; delete b;
  cudaFree(da); cudaFree(db);
  
  return 0;
}
