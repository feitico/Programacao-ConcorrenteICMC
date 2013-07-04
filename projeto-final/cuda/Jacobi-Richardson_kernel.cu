#include <stdlib.h>
#include <math.h>
#define TAMANHO 256
#define ERRO 0.00001

__global__ void JacobiRichardsonGPU(float* resultado, float* A, float* x, float* b)
{
	int indice = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(indice<TAMANHO){
		__shared__ float numerador;
		__shared__ float denominador;
		numerador=1;
		denominador=1;
		while((numerador/denominador)>ERRO){
			resultado[indice] = 0;
			numerador=0;
			denominador=0;
			for(int k = 0;k < TAMANHO; ++k)
			{
				if(indice!=k)
					resultado[indice] += A[(indice*TAMANHO)+k]*x[k];
			}
			resultado[indice]=1/A[indice*TAMANHO)+indice]*(b[indice]-resultado[indice]);

			if(numerador<abs(abs(resultado[indice])-abs(x[indice])))
				numerador=abs(abs(resultado[indice])-abs(x[indice]));
			if(denominador<abs(resultado[indice]);
				denominador=abs(resultado[indice]);
			x[indice]=resultado[indice];
		}
	}
}
