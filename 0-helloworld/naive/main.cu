#include <iostream>

__global__ void mykernel(void){
}

int main(int argc, char* argv[]){
  mykernel<<<1, 1>>>();
  std::cout << "Hello World!" << std::endl;
  return 0;
}
