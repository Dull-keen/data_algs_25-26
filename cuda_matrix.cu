#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 2
#define PRINT_PART_SIZE 2 

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
    size_t bytes = N * N * sizeof(float);

    // Выделение и инициализация матриц на CPU
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

    print_matrix_part("A (part)", A);
    print_matrix_part("B (part)", B);


auto start = std::chrono::high_resolution_clock::now();
matmul_cpu(A, B, C_cpu);
auto end = std::chrono::high_resolution_clock::now();
double cpu_time = std::chrono::duration<double>(end - start).count();

printf("CPU time: %.3f seconds\n", cpu_time);


    
    // Выделение памяти на GPU
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

              
    auto gstart = std::chrono::high_resolution_clock::now();
    matmul_gpu<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    auto gend = std::chrono::high_resolution_clock::now();

    double gpu_time = std::chrono::duration<double>(gend - gstart).count();

    printf("GPU time: %.3f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost);

    print_matrix_part("C_cpu (part)", C_cpu);
    print_matrix_part("C_gpu (part)", C_gpu);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A); free(B); free(C_cpu); free(C_gpu);
}
