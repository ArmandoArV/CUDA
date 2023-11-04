#include "cuda_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void cuadratica(float a, float b, float c, float* d, float* res1, float* res2)
{
    *d = pow(b, 2) - (4 * a * c);

    if (*d >= 0) {
        *res1 = (-b + sqrt(*d)) / (2 * a);
        *res2 = (-b - sqrt(*d)) / (2 * a);
    }
    else {
        *res1 = *res2 = 0.0f;
    }
}

__host__ void cuadraticCPU(float a, float b, float c, float* x1, float* x2) {
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
        *x1 = (-b + sqrt(discriminant)) / (2 * a);
        *x2 = (-b - sqrt(discriminant)) / (2 * a);
    }
    else {
        *x1 = *x2 = 0.0f;
    }
}

int main(int argc, char** argv)
{
    float a = 1, b = 3, c = 2;
    float hst_x1, hst_x2;
    float* dev_c;
    float* dev_res1;
    float* dev_res2;

    // Allocate memory for device variables
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_c, sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_res1, sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_res2, sizeof(float));

    // Calculate the result on the CPU
    cuadraticCPU(a, b, c, &hst_x1, &hst_x2);

    printf("Funcion cuadratica: %f x^2 + %f x + %f\n", a, b, c);
    printf("Resultado CPU (x1): %f\n", hst_x1);
    printf("Resultado CPU (x2): %f\n", hst_x2);

    // Launch the GPU kernel
    cuadratica << <1, 1 >> > (a, b, c, dev_c, dev_res1, dev_res2);

    // Copy GPU results back to host
    float hst_res1, hst_res2;
    cudaStatus = cudaMemcpy(&hst_res1, dev_res1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(&hst_res2, dev_res2, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultado GPU (x1): %f\n", hst_res1);
    printf("Resultado GPU (x2): %f\n", hst_res2);

    // Free allocated memory
    cudaFree(dev_c);
    cudaFree(dev_res1);
    cudaFree(dev_res2);

    return 0;
}
