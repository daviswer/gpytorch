#include <THC/THC.h>

__global__ void ComplexPointwiseMulAndScale (cufftComplex *a, cufftComplex *b, int size) { 
  const int numThreads = blockDim.x * gridDim.x; 
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x; 
  float scale = 1.0f / (float)size; 
  cufftComplex c; 
  for (int i = threadID; i < size; i += numThreads) { 
    c = cuCmulf(a[i], b[i]); b[i] = make_cuFloatComplex(scale*cuCrealf(c), scale*cuCimagf(c)); 
  } 
  return; 
}

