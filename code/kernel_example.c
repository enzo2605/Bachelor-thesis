#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel definition
__global__ void VecAdd(float *A, float *B, float *C) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    C[index] = A[index] + B[index];
}

int main() {
    ...
    // Set up the parameters of the kernel
    dim3 nThreadsPerBlock, nBlocks;
    nBlocks.x = 3;
    nThreadsPerBlock.x = N / nBlocks.x + ((N % nBlocks.x == 0) ? 0 : 1);
    // Kernel invocation with N threads
    VecAdd <<< nBlocks, nThreadsPerBlock >>> (A, B, C);
    ...
}