#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <time.h>
#include <string.h>

#define N 1000 // Matrix size N x N
#define PRINT_PART_SIZE 5 // Size of the matrix part to print 

void print_matrix_part(const char* name, float* M)
{
    printf("%s:\n", name);
    for (int i = 0; i < PRINT_PART_SIZE; i++) 
    {
        for (int j = 0; j < PRINT_PART_SIZE; j++) 
        {
            printf("%6.1f ", M[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matmul_cpu(float* A, float* B, float* C) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++)  
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) 
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float* A, float* B, float* C) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) 
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) 
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main() 
{
    srand((unsigned int)time(NULL));
    size_t bytes = N * N * sizeof(float);

    // Memory allocation and initialization on CPU
    float *A, *B, *C_cpu, *C_gpu;
    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C_cpu = (float*)malloc(bytes);
    C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) 
    {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
    }

    // Printing parts of matrices A and B
    print_matrix_part("A (part)", A);
    print_matrix_part("B (part)", B);

    // Memory allocation on GPU
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

              
    printf("Launching GPU kernel...\n");

    auto gstart = std::chrono::high_resolution_clock::now();
    matmul_gpu<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    auto gend = std::chrono::high_resolution_clock::now();

    double gpu_time = std::chrono::duration<double>(gend - gstart).count();

    printf("GPU time: %.3f seconds\n", gpu_time);

    // Copy back to CPU
    cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost);
    print_matrix_part("C_gpu (part)", C_gpu);


    printf("Computing on CPU...\n");

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();

    printf("CPU time: %.3f seconds\n", cpu_time);

    print_matrix_part("C_cpu (part)", C_cpu);
    
    printf("GPU speedup: %.2fx\n", cpu_time / gpu_time);

    printf("Verification: %s\n", (memcmp(C_cpu, C_gpu, bytes) == 0) ? "SUCCESS" : "FAILURE");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A); free(B); free(C_cpu); free(C_gpu);
}
