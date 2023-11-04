// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void triangleArea(float base, float altura, float* res)
{
	*res = (base * altura)/2;
}



__host__ float triArea(float base, float altura)
{
	return (base * altura)/2;
}

int main(int argc, char** argv)
{
	float base = 52, altura = 3, c=0;
	float* hst_c;
	float* dev_c;


	hst_c = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&dev_c, sizeof(float));


	c = triArea(base, altura);
	printf("CPU:");
	printf("Area del triangulo: %f\n", c);

	triangleArea<<<1,1>>>(base, altura, dev_c);

	cudaMemcpy(hst_c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);
	printf("GPU: %f\n", *hst_c);


	free(hst_c);
	cudaFree(dev_c);
	return 0;

}