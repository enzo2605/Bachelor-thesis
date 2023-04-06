__host__
void f_PeerClassicTwoStagesGPU(int N, Vector t_span, Vector y0, Matrix L, Vector *yT, Matrix *y, Vector *t, int nThreadsPerBlockX, int nThreadsPerBlockY, int nBlocksX, int nBlocksY) {
    /******************************* 
    * Fixing method coefficients 
    * ****************************/
    double b11 = 0.0f, b21 = 0.0f, b12 = 1.0f, b22 = 1.0f, c1 = 0.0f, c2 = 1.0f, r21 = 0.0f;

    double a11 = -((b11 - 2 * b11 * c1 - pow(c1, 2) + b11 * pow(c1, 2)) / (2 * (-1 + c1)));
    double a12 = -((b11 + 2 * c1 - 2 * b11 * c1 - pow(c1, 2) + b11 * pow(c1, 2)) / (2 * (-1 + c1)));
    double a21 = -((-1 + b21 - 2 * b21 * c1 + b21 * pow(c1, 2) + 2 * c1 * r21) / (2 * (-1 + c1)));
    double a22 = -((3 + b21 - 2 * c1 - 2 * b21 * c1 + b21 * pow(c1, 2) - 2 * r21) / (2 * (-1 + c1)));
    //fprintf(stdout, "a11: %f\na12: %f\na21: %f\na22: %f\n", a11, a12, a21, a22);

    double c[STAGES] = { c1, c2 };
    //printDVector(c, STAGES, "c");

    Matrix A;
    A.elements = (double *)Calloc(STAGES * STAGES, sizeof(double));
    double tempA[STAGES * STAGES] = { a11, a12, a21, a22 };
    initMatrixByRowWithValuesFromVector(A.elements, STAGES, STAGES, tempA, STAGES * STAGES);
    //printDMatrix(A, STAGES, STAGES, "A");

    Matrix B;
    B.elements = (double *)Calloc(STAGES * STAGES, sizeof(double));
    double tempB[STAGES * STAGES] = { b11, b12, b21, b22 };
    initMatrixByRowWithValuesFromVector(B.elements, STAGES, STAGES, tempB, STAGES * STAGES);
    //printDMatrix(B, STAGES, STAGES, "B");

    Matrix R;
    R.elements = (double *)Calloc(STAGES * STAGES, sizeof(double));
    double tempR[STAGES * STAGES] = { 0.0f, 0.0f, r21, 0.0f };
    initMatrixByRowWithValuesFromVector(R.elements, STAGES, STAGES, tempR, STAGES * STAGES);
    //printDMatrix(R, STAGES, STAGES, "R");

    /******************************* 
    *  Compute the solution
    * ****************************/
    double h = (t_span.elements[1] - t_span.elements[0]) / N;
    t->elements = linspace(t_span.elements[0], t_span.elements[1], N + 1);
    t->dim = N + 1;
    // Initialize conditions
    int k = 0;
    int n = 1;
    int s = STAGES; // Number of stages
    int problemDimension = y0.dim; // Dimension of the problem

    // Host data
    Vector host_FYi, host_FYiRK, host_Yi;
    Matrix host_Y;
    host_FYi.dim = host_FYiRK.dim = host_Yi.dim = yT->dim = problemDimension;
    y->rows = problemDimension; y->cols = N + 1;
    host_FYi.sizeInBytes = host_FYiRK.sizeInBytes = host_Yi.sizeInBytes = problemDimension * sizeof(double);
    host_Y.rows = s * problemDimension; host_Y.cols = (N / N_CHUNK) + 1;
    // Host data allocation
    host_FYi.elements   = (double *)Calloc(host_FYi.dim, sizeof(double));
    host_FYiRK.elements = (double *)Calloc(host_FYiRK.dim, sizeof(double));
    host_Yi.elements    = (double *)Calloc(host_Yi.dim, sizeof(double));
    yT->elements        = (double *)Calloc(yT->dim, sizeof(double));
    y->elements         = (double *)Calloc(y->rows * y->cols, sizeof(double));
    host_Y.elements     = (double *)Calloc(host_Y.rows * host_Y.cols, sizeof(double));

    // Device data
    Vector dev_FYi, dev_FYiRK, dev_Yi, dev_yT, dev_y0, dev_Fnm1, dev_temp;
    Matrix dev_Y, dev_y, dev_A, dev_B;
    // Initialize dimension
    dev_FYi.dim = dev_FYiRK.dim = dev_Yi.dim = dev_yT.dim = dev_y0.dim = problemDimension;
    dev_Fnm1.dim = dev_temp.dim = s * problemDimension;
    dev_Y.rows = s * problemDimension; dev_Y.cols = (N / N_CHUNK) + 1;
    dev_y.rows = problemDimension; dev_y.cols = N + 1;
    dev_A.rows = dev_A.cols = dev_B.rows = dev_B.cols = s;
    // Initialize size in byte
    dev_FYi.sizeInBytes = dev_FYiRK.sizeInBytes = dev_Yi.sizeInBytes = dev_yT.sizeInBytes = problemDimension * sizeof(double);
    dev_y0.sizeInBytes = problemDimension * sizeof(double);
    dev_Fnm1.sizeInBytes = dev_Fnm1.dim * sizeof(double);
    dev_temp.sizeInBytes = dev_temp.dim * sizeof(double);
    dev_Y.sizeInBytes = dev_Y.rows * dev_Y.cols * sizeof(double);
    dev_y.sizeInBytes = dev_y.rows * dev_y.cols * sizeof(double);
    dev_A.sizeInBytes = dev_A.rows * dev_A.cols * sizeof(double);
    dev_B.sizeInBytes = dev_B.rows * dev_B.cols * sizeof(double);
    // Device data allocation
    checkCudaErrors( cudaMalloc((void **)&dev_FYi.elements, dev_FYi.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_FYiRK.elements, dev_FYiRK.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_Fnm1.elements, dev_Fnm1.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_temp.elements, dev_temp.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_Yi.elements, dev_Yi.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_Y.elements, dev_Y.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_y.elements, dev_y.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_yT.elements, dev_yT.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_y0.elements, dev_y0.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_A.elements, dev_A.sizeInBytes) );
    checkCudaErrors( cudaMalloc((void **)&dev_B.elements, dev_B.sizeInBytes) );
    // Device data initialization
    checkCudaErrors( cudaMemset(dev_Fnm1.elements, 0x0000, dev_Fnm1.sizeInBytes) );
    checkCudaErrors( cudaMemset(dev_Yi.elements, 0x0000, dev_Yi.sizeInBytes) );
    checkCudaErrors( cudaMemset(dev_Y.elements, 0x0000, dev_Y.sizeInBytes) );
    checkCudaErrors( cudaMemset(dev_y.elements, 0x0000, dev_y.sizeInBytes) );
    checkCudaErrors( cudaMemset(dev_yT.elements, 0x0000, dev_yT.sizeInBytes) );
    checkCudaErrors( cudaMemcpy(dev_y0.elements, y0.elements, dev_y0.sizeInBytes, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_A.elements, A.elements, dev_A.sizeInBytes, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_B.elements, B.elements, dev_B.sizeInBytes, cudaMemcpyHostToDevice) );

    /************************************************
    * Runge-Kutta of order four to initialize stages 
    * **********************************************/
    dim3 nBlocks(nBlocksX, nBlocksY), nThreadsPerBlock(nThreadsPerBlockX, nThreadsPerBlockY);
    printf("\nf_PeerClassicTwoStagesGPU\nnBlocks(%d, %d)\tnThreadsPerBlock(%d, %d)\n", nBlocks.x, nBlocks.y, nThreadsPerBlock.x, nThreadsPerBlock.y);
    
    // Compute Y first column
    k = 0;
    for (int i = 0; i < s; i++) {
        // Apply Runge-Kutta 4th to y0
        host_FYiRK.elements = RungeKutta4th(c[i] * h, t->elements[0], y0.elements, y0.dim, L.elements, L.rows, &host_FYiRK.dim);
        checkCudaErrors( cudaMemcpy(dev_FYiRK.elements, host_FYiRK.elements, host_FYiRK.sizeInBytes, cudaMemcpyHostToDevice) );
        // Copy the resulting function of Runge-Kutta in the first column of matrix Y
        computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_Y.elements[k * dev_Y.cols + (n - 1)], problemDimension, 1, dev_Y.cols, dev_FYiRK.elements, 1);
        k += problemDimension;
    }

    // Compute y first column
    computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (dev_y.elements, problemDimension, 1, dev_y.cols, dev_y0.elements, 1);
    // Compute y second column
    computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_y.elements[0 * (dev_y.cols) + n], problemDimension, 1, dev_y.cols, &dev_Y.elements[problemDimension * dev_Y.cols + 0], dev_Y.cols);

    // Solution at t0+cs*h (cs=1)
    k = 0;
    for (int i = 0; i < s; i++) {
        // Copy half column of Y in Yi
        computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (dev_Yi.elements, problemDimension, 1, 1, &dev_Y.elements[k * dev_Y.cols + (n - 1)], dev_Y.cols);
        // Move Yi from device to host
        checkCudaErrors( cudaMemcpy(host_Yi.elements, dev_Yi.elements, dev_Yi.sizeInBytes, cudaMemcpyDeviceToHost) );
        // Apply Sherratt function to Yi and obtain FYi
        host_FYi.elements = Sherratt(host_Yi.elements, problemDimension, L.elements, L.rows, &host_FYi.dim);
        // Move FYi from host to device
        checkCudaErrors( cudaMemcpy(dev_FYi.elements, host_FYi.elements, host_FYi.sizeInBytes, cudaMemcpyHostToDevice) );
        // Copy the result from Sherratt in Fnm1
        computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_Fnm1.elements[k], problemDimension, 1, 1, dev_FYi.elements, 1);
        k += problemDimension;
    }

    int chunkCounter = 0;
    int colsCounter = 1;
    for (n = 1; n < N; n++) {
        //printf("\nn: %d\n", n);
        if (colsCounter == dev_Y.cols) {
            colsCounter = 1;
            chunkCounter++;

            dim3 nThreadsPerBlock2(nThreadsPerBlockX, nThreadsPerBlockX), nBlocks2(N_BLOCKS(nThreadsPerBlockX, dev_Y.rows));
            // Copy the last column of Y into the temp vector
            computeOneColumnGPU <<< nBlocks2, nThreadsPerBlock2 >>> (dev_temp.elements, dev_Y.rows, 1, 1, &dev_Y.elements[0 * dev_Y.cols + (dev_Y.cols - 1)], dev_Y.cols);
            // Clear the Y memory
            checkCudaErrors( cudaMemset(dev_Y.elements, 0x0000, dev_Y.rows * dev_Y.cols * sizeof(double)) );
            // Copy the vector temp into the first column of Y
            computeOneColumnGPU <<< nBlocks2, nThreadsPerBlock2 >>> (dev_Y.elements, dev_Y.rows, 1, dev_Y.cols, dev_temp.elements, 1);
            // Barrier for the threads
            checkCudaErrors( cudaDeviceSynchronize() );
        }
        
        for (int i = 0; i < s; i++) {
            // Kernel 1
            computeYGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_Y.elements[(i * problemDimension) * dev_Y.cols + colsCounter], 
                                                            &dev_Y.elements[colsCounter - 1], problemDimension, 1, dev_Y.cols, 
                                                            dev_A.elements, dev_B.elements, 2, i, 0,
                                                            dev_Fnm1.elements, 1, h);
            // Kernel 2
            computeYGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_Y.elements[(i * problemDimension) * dev_Y.cols + colsCounter], 
                                                            &dev_Y.elements[(problemDimension) * dev_Y.cols + (colsCounter - 1)], problemDimension, 1, dev_Y.cols, 
                                                            dev_A.elements, dev_B.elements, 2, i, 1, 
                                                            &dev_Fnm1.elements[problemDimension], 1, h);
            checkCudaErrors( cudaDeviceSynchronize() );
        }

        // Clear the dev_Fnm1 and dev_Yi memory
        checkCudaErrors( cudaMemset(dev_Fnm1.elements, 0x0000, dev_Fnm1.sizeInBytes) );
        checkCudaErrors( cudaMemset(dev_Yi.elements, 0x0000, dev_Yi.sizeInBytes) );
        // The index k will point to the correct half of Y column
        k = 0;
        for (int i = 0; i < s; i++) {
            // Copy half column of Y in Yi
            computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (dev_Yi.elements, problemDimension, 1, 1, &dev_Y.elements[k * dev_Y.cols + colsCounter], dev_Y.cols);
            // Move Yi from device to host
            checkCudaErrors( cudaMemcpy(host_Yi.elements, dev_Yi.elements, dev_Yi.sizeInBytes, cudaMemcpyDeviceToHost) );
            // Apply Sherratt function to Yi and obtain FYi
            host_FYi.elements = Sherratt(host_Yi.elements, problemDimension, L.elements, L.rows, &host_FYi.dim);
            // Move FYi from host to device
            checkCudaErrors( cudaMemcpy(dev_FYi.elements, host_FYi.elements, host_FYi.sizeInBytes, cudaMemcpyHostToDevice) );
            // Copy the result from Sherratt in Fnm1
            computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_Fnm1.elements[k], problemDimension, 1, 1, dev_FYi.elements, 1);
            k += problemDimension;
            checkCudaErrors( cudaDeviceSynchronize() );
        }

        // Copy the n-th column of Y in the n+1-th column of y
        computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (&dev_y.elements[0 * (dev_y.cols) + (n + 1)], problemDimension, 1, dev_y.cols, 
                                                                 &dev_Y.elements[problemDimension * dev_Y.cols + colsCounter], dev_Y.cols);
        checkCudaErrors( cudaDeviceSynchronize() );
        
        colsCounter++;
    }

    // Compute yT by coping the last column of y in it
    computeOneColumnGPU <<< nBlocks, nThreadsPerBlock >>> (dev_yT.elements, problemDimension, 1, 1, &dev_y.elements[N], dev_y.cols);

    // Copy final result from device to host
    checkCudaErrors( cudaMemcpy(y->elements, dev_y.elements, dev_y.sizeInBytes, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy(yT->elements, dev_yT.elements, dev_yT.sizeInBytes, cudaMemcpyDeviceToHost) );

    // Free the host and device memory allocated
    free(host_FYi.elements);
    free(host_FYiRK.elements);
    free(host_Yi.elements);
    free(host_Y.elements);
    checkCudaErrors( cudaFree(dev_FYi.elements)   );
    checkCudaErrors( cudaFree(dev_FYiRK.elements) );
    checkCudaErrors( cudaFree(dev_Fnm1.elements)  );
    checkCudaErrors( cudaFree(dev_temp.elements)  );
    checkCudaErrors( cudaFree(dev_Yi.elements)    );
    checkCudaErrors( cudaFree(dev_Y.elements)     );
    checkCudaErrors( cudaFree(dev_y.elements)     );
    checkCudaErrors( cudaFree(dev_yT.elements)    );
    checkCudaErrors( cudaFree(dev_y0.elements)    );
    checkCudaErrors( cudaFree(dev_A.elements)     );
    checkCudaErrors( cudaFree(dev_B.elements)     );
}