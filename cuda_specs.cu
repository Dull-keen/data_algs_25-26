#include <stdio.h>
#include <cuda_runtime.h>

int main() 
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) 
    {
        printf("cudaGetDeviceCount error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d:\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
    }

    return 0;
}