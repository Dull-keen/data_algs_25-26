#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
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

void matmul(float* A, float* B, float* C) 
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

void matmul_omp(float* A, float* B, float* C) 
{
    #pragma omp parallel for num_threads(8)
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

int main() 
{
    srand((unsigned int)time(NULL));
    size_t bytes = N * N * sizeof(float);

    // Memory allocation and initialization
    float *A, *B, *C, *C_omp;
    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C = (float*)malloc(bytes);
    C_omp = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) 
    {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
    }

    print_matrix_part("A (part)", A);
    print_matrix_part("B (part)", B);


    printf("Computing on CPU with OpenMP...\n");
    
    double start_omp = omp_get_wtime();
    matmul_omp(A, B, C_omp);
    double end_omp = omp_get_wtime();
    double omp_time = end_omp - start_omp;
    
    printf("OpenMP time: %.3f seconds\n", omp_time);
    print_matrix_part("C_omp (part)", C_omp);
    

    printf("Computing on CPU alone...\n");

    double start_seq = omp_get_wtime();
    matmul(A, B, C);
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;
    
    printf("Sequential time: %.3f seconds\n", seq_time);
    print_matrix_part("C (part)", C);
    
    printf("OpenMP speedup: %.2fx\n", seq_time / omp_time);

    printf("Verification: %s\n", (memcmp(C, C_omp, bytes) == 0) ? "SUCCESS" : "FAILURE");

    free(A); free(B); free(C); free(C_omp);
}
