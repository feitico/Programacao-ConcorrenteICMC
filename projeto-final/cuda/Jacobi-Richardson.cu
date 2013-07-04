#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <Jacobi-Richardson_kernel.cu>

void InicializarValores(float*,float*,float*);
void InicializarValores2(float*,float*,float*);
void InicializarValores3(float*,float*,float*);
bool TestarConvergencia(float*);
void JacobiRichardsonCPU(float*,float*,float*,float*,int,float);
	

int main(int argc, char** argv)
{
	if(cutCheckCmdLineFlag(argc,(const char**)argv, "device"))
		cutilDeviceInit(argc,argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId());

	srand(20);

	int size_A = TAMANHO * TAMANHO;
	int mem_size_A = sizeof(float)*size_A;
	float* h_A = (float*) malloc(mem_size_A);

	int size_x = TAMANHO;
	int mem_size_x = sizeof(float)*size_x;
	float* h_x = (float*) malloc(mem_size_x);

	int size_b = TAMANHO;
	int mem_size_b = sizeof(float)*size_b;
	float* h_b = (float*) malloc(mem_size_b);

	int size_y = TAMANHO;
	int mem_size_y = sizeof(float)*size_y;
	float* h_y = (float*) malloc(mem_size_y);

	InicializarValores2(h_A,h_x,h_b);

	if(TestarConvergencia(H_A))
	{
		float* d_A;
		cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
		float* d_x;
		cutilSafeCall(cudaMalloc((void**) &d_x, mem_size_x));
		float* d_b;
		cutilSafeCall(cudaMalloc((void**) &d_b, mem_size_b));
		float* d_y;
		cutilSafeCall(cudaMalloc((void**) &d_y, mem_size_y));
	
		cutilSafeCall(cudaMemcpy(d_A,h_A,mem_size_A,cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_x,h_x,mem_size_x,cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_b,h_b,mem_size_b,cudaMemcpyHostToDevice));

		int blocos = TAMANHO/max_threas_por_bloco;
		unsigned int timer = 0;
		cutilCheckError(cutCreateTimer(&timer));
		cutilCheckError(cutStartTimer(timer));
		JacobiRichardsonGPU<<<blocos,max_threads_por_bloco>>>(d_y,d_A,d_x,d_b);
		
		cutilCheckError(cutStopTimer(timer));
		printf("Tempo de processamento da GPU: %f ms. \n", cutGetTimerValue(timer));
		cutilCheckError(cutDeleteTimer(timer));

		cudaMemcpy(h_y_GPU,d_y,mem_size_y,cudaMemcpyDeviceToHost);
		unsigned int timer2 = 0;
		cutilCheckError(cutCreateTimer(&timer2));
		cutilCheckError(cutStartTimer(timer2));
		JacobiRichardsonCPU(h_y_CPU,h_A,h_x,h_b,TAMANHO,ERRO);
		cutilCheckError(cutStopTimer(timer2));
		printf("Tempo de processamento da CPU: %f ms. \n", cutGetTimerValue(timer2));
		cutilCheckError(cutDeleteTimer(timer2));

		for(int h=0;h<TAMANHO;h++)
			printf("%f",h_y_GPU[h]);
			printf("\n");

		for(int h=0;h<TAMANHO;h++)
			printf("%f",h_y_CPU[h]);
			printf("\n");

		free(h_A);
		free(h_b);
		free(h_y_GPU);
		free(h_x);
		free(h_y_CPU);
		cudaFree(d_A);
		cudaFree(d_b);
		cudaFree(d_x);
		cudaFree(d_y);

	}
	else
		printf("Sistema nao converge");
	cudaThreadExit();
	cutilExit(argc,argv);
}		

void InicializarValores(float *h_A,float *h_x,float *h_b)
{
	for(int i=0;i<TAMANHO;i++)
	{
		for(int j=0;j<TAMANHO;j++){
			if(i!=j)
			{
				h_A[i*TAMANHO+j]=1;
			}
			else
			{
				h+A[i*TAMANHO+j]=j+TAMANHO+60;
			}
		}
	h_x[i]=TAMANHO+90;
	h_b[i]=TAMANHO+95;
	}
}

bool TestarConvergencia(float *h_A)
{
	float linha_soma,coluna_soma;
	for(int i=0;i<TAMANHO;i++)
	{
		linha_soma=0;
		for(int j=0;j<TAMANHO;j++)
			if(i!=j)
				linha_soma += h_A[j*TAMANHO+i];
		linha_soma = linha_soma/h_A[i*TAMANHO+i];
		if(linha_soma>1)
		{
			for(int i=0;i<TAMANHO;i++)
			{
				coluna_soma=0;
				for(int j=0;j<TAMANHO;j++)
					if(i!=j)
						coluna_soma += h_A[j*TAMANHO+i];
				coluna_soma = coluna_soma/h_A[i*TAMANHO+i];

				if(coluna_soma>1)
					return false;
			}
			return true;
		}
	}
	
	return true;
}
