#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// decomposes n into a sum of squares and prints the result
void print_decompose_squares(int n) 
{
    if (n <= 0) 
    {
        printf("undefined for n <= 0");
        return;
    }
    int rem = n;
    int first = 1;
    while (rem > 0)
    {
        int root = (int)sqrt((double)rem);
        int sq = root * root;
        if (!first) 
        {
            printf(" + ");
        }
        printf("%d^2", root);
        rem -= sq;
        first = 0;
    }
}

// Fibonacci number F(n)
unsigned long long fib(unsigned int n) 
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    unsigned long long a = 0, b = 1, c;
    for (unsigned int i = 2; i <= n; ++i)
    {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Check if x is prime
int is_prime(int x) 
{
    if (x < 2) return 0;
    if (x == 2) return 1;
    if (x % 2 == 0) return 0;
    int lim = (int)sqrt((double)x);
    for (int i = 3; i <= lim; i += 2) 
    {
        if (x % i == 0) return 0;
    }
    return 1;
}

// n-th prime number
int nth_prime(int n) 
{
    if (n <= 0) return -1;
    int count = 0;
    int candidate = 1;
    while (count < n) 
    {
        ++candidate;
        if (is_prime(candidate)) 
        {
            ++count;
        }
    }
    return candidate;
}

// sum of all divisors of n
long long sum_divisors(int n) 
{
    if (n == 0) return 0;
    if (n < 0) n = -n;
    long long sum = 0;
    int lim = (int)sqrt((double)n);
    for (int i = 1; i <= lim; ++i) 
    {
        if (n % i == 0) 
        {
            sum += i;
            int j = n / i;
            if (j != i) sum += j;
        }
    }
    return sum;
}



int main(int argc, char *argv[]) 
{
    if (argc < 2) 
    {
        fprintf(stderr, "Usage: %s input.txt\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) 
    {
        perror("fopen");
        return 1;
    }

    /*mb better to read from file in critical section without memory allocation, but I doubt it*/
    // read integers from file into dynamic array
    int capacity = 5;
    int *a = (int *)malloc(capacity * sizeof(int));
    if (!a) 
    {
        perror("malloc");
        fclose(f);
        return 1;
    }
    
    int x;
    int size = 0;
    while (fscanf(f, "%d", &x) == 1) 
    {
        if (size == capacity) 
        {
            capacity *= 2;
            int *tmp = (int *)realloc(a, capacity * sizeof(int));
            if (!tmp) 
            {
                perror("realloc");
                free(a);
                fclose(f);
                return 1;
            }
            a = tmp;
        }
        a[size++] = x;
    }
    fclose(f);

    // Shared index for the next number to process
    int next_idx = 0;

    #pragma omp parallel sections shared(a, size, next_idx)
    {
        // SECTION 1: Decomposition into sum of squares
        #pragma omp section
        {
            for (;;) 
            {
                int my_idx;
                // Critical section to safely get the next index
                #pragma omp critical(queue)
                {
                    if (next_idx < size) 
                    {
                        my_idx = next_idx;
                        next_idx++;
                    } else 
                    {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                /*#pragma omp atomic (optional but not ideal)*/
                //{
                printf("1st section> a[%d] = %d: ", my_idx, n);
                print_decompose_squares(n); // prints could separate 
                printf("\n");
                //}
            }
        }

        // SECTION 2: Fibonacci number F(n)
        #pragma omp section
        {
            for (;;) 
            {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < size) 
                    {
                        my_idx = next_idx;
                        next_idx++;
                    } else 
                    {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                if (n < 0) 
                {
                    printf("2nd section> a[%d] = %d: Fibonacci undefined for n < 0\n", my_idx, n);
                } else 
                {
                    unsigned long long f = fib((unsigned int)n);
                    printf("2nd section> a[%d] = %d: F(%d) = %llu\n", my_idx, n, n, f);
                }
            }
        }

        // SECTION 3: n-th prime number
        #pragma omp section
        {
            for (;;) 
            {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < size) 
                    {
                        my_idx = next_idx;
                        next_idx++;
                    } else 
                    {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                if (n <= 0) 
                {
                    printf("3rd section> a[%d] = %d: nth prime undefined for n <= 0\n", my_idx, n);
                } else 
                {
                    int p = nth_prime(n);
                    printf("3rd section> a[%d] = %d: %d-th prime = %d\n", my_idx, n, n, p);
                }
            }
        }

        // SECTION 4: sum of all divisors of n
        #pragma omp section
        {
            for (;;) 
            {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < size) 
                    {
                        my_idx = next_idx;
                        next_idx++;
                    } else 
                    {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                long long s = sum_divisors(n);
                printf("4th section> a[%d] = %d: sum of divisors = %lld\n", my_idx, n, s);
            }
        }
    }

    free(a);
    return 0;
}
