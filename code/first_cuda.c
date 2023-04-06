#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void initializeArray(int *array, int n);
void stampaArray(int* array, int n);
void equalArray(int *a, int *b, int n);

int main(int argn, char **argv) {
    int N; // total number of array elements
    int *A_host; // array of host
    int *A_device; // array of the device
    int *copy; // array for copy data from the device
    int size; // size in byte of each array
    
    if (argn == 1) {
        N = 20;
    } else {
        N = atoi(argv[1]);
        printf("**********\tFIRST EXAMPLE\t**********\n");
        printf("copy of %d elements from CPU to GPU and vice versa\n\n", N);
    }

    // size in byte for each array
    size = N * sizeof(int);

    // host data allocation
    A_host = (int *)malloc(size);
    if (A_host == NULL) {
        printf("Malloc error.\n");
        exit(EXIT_FAILURE);
    }
    copy = (int *)malloc(size);
    if (copy == NULL) {
        printf("Malloc error.\n");
        exit(EXIT_FAILURE);
    }

    // device data allocation
    cudaMalloc((void **)&A_device, size);
    // host data initialization
    initializeArray(A_host, N);

    // copy data from host to device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    // copy result from device to host
    cudaMemcpy(copy, A_device, size, cudaMemcpyDeviceToHost);

    printf("array of host\n");
    stampaArray(A_host,N);
    printf("array copied from device\n");
    stampaArray(copy,N);
    //accuracy test
    equalArray(copy, A_host,N);
    //host data de-allocation
    free(A_host);
    free(copy);
    //device data de-allocation
    cudaFree(A_device);

    exit(0);
}

void initializeArray(int *array, int n) {
    int i;
    for (i = 0; i < n; i++)
        array[i] = i;
}

void stampaArray(int* array, int n) {
    int i;
    for(i = 0; i < n; i++)
        printf("%d ", array[i]);
    printf("\n");
}

void equalArray(int *a, int *b, int n) {
    int i = 0;
    while (a[i] == b[i])
        i++;
    if (i < n)
        printf(" The results of the host and the device are different \n");
    else
        printf(" The results of the host and the device are the same \n");
}