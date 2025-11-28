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
        double total_mem_gib = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("  Total global memory: %.2f GB (or %zu bytes)\n", total_mem_gib, prop.totalGlobalMem); 
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }

    return 0;
}