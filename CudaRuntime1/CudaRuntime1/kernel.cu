// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// declaracion de funciones
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma_GPU(int a, int b, int* c)
{
	*c = a + b;
}
// HOST: funcion llamada y ejecutada desde el host
__host__ int suma_CPU(int a, int b)
{
	return (a + b);
}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	int n1 = 1, n2 = 2, c = 0;
	int* hst_c;
	int m1 = 10, m2 = 20;
	int* dev_c;
	// reserva en el host
	hst_c = (int*)malloc(sizeof(int));
	// reserva en el device
	cudaMalloc((void**)&dev_c, sizeof(int));
	// llamada a la funcion suma_CPU
	c = suma_CPU(n1, n2);
	// resultados CPU
	printf("CPU:\n");
	printf("%2d + %2d = %2d \n", n1, n2, c);
	// llamada a la funcion suma_GPU
	suma_GPU << <1, 1 >> > (m1, m2, dev_c);
	// recogida de datos desde el device
	cudaMemcpy(hst_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	// resultados GPU
	printf("GPU:\n");
	printf("%2d + %2d = %2d \n", m1, m2, *hst_c);
	// salida
	printf("\npulsa INTRO para finalizar...");
	fflush(stdin);
	char tecla = getchar();
	return 0;
}