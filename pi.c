#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>

long double compute_pi(unsigned long long n)
{
    long double h = 1.0 / (double)n;
    long double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) num_threads(8)
    for (long long i = 0; i < n; ++i) 
    {
        long double x = (i + 0.5) * h;
        sum += sqrt(1.0 - x * x);
    }

    long double quarter_area = sum * h;
    return 4.0 * quarter_area;
}

int main()
{
    long double eps;
    printf("Enter eps value: ");
    if (scanf("%Lf", &eps) != 1 || eps <= 0.0) 
    {
        printf("Invalid input for eps.\n");
        return 1;
    }

    unsigned long long n = 100; // start with 100 squares
    long double pi_prev = 3.0;
    long double pi_curr = compute_pi(n);

    double start_omp = omp_get_wtime();
    while (fabs(pi_curr - pi_prev) > eps)
    {
        pi_prev = pi_curr;
        n *= 2;
        pi_curr = compute_pi(n);
    }
    double end_omp = omp_get_wtime();
    double omp_time = end_omp - start_omp;

    printf("      Known pi = 3.14159265358979323846\n");
    printf(" Calculated pi = %.20Lf\n", pi_curr);
    printf("                  ^^^^^^^^^^^^^^^^^^^^^\n");
    printf("Counter---------> 012345678901234567890\n\n");

    printf("Number of squares n = %llu\n", n);
    printf("OpenMP time: %.3f seconds\n\n", omp_time);

    long double pi_true = 3.141592653589793238462643383279502884L;
    printf("Abs diff with known pi = %.4Le\n", fabsl(pi_curr - pi_true));
    return 0;
}
