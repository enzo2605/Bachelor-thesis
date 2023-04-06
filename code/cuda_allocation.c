#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define M 10
#define N 4
int main(int argn, char **argv) {
    double *a;
    a = (double *)malloc(M * N * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Errore allocazione.\n");
        exit(EXIT_FAILURE);
    }
    exit(0);
}