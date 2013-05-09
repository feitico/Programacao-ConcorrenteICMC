#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_NUM 8000
/*#define RAIZ_MAX_NUM 90*/

int ehPrimo[MAX_NUM];

/* processa somente os numeros primos de um bloco especifico */
int crivo_singleBlock(const int de, const int para){
	printf("valor de 'de': %d e 'para': %d\n", de, para);	
	const int tamanhoMemoria = ( para - de + 1 ) / 2;
	printf("valor de tamanhoMemoria: %d\n", tamanhoMemoria);	

	int i, j, minJ, indice, encontrou;

	/*inicializacao*/
	/*char * ehPrimo;
	ehPrimo = (char*) malloc (tamanhoMemoria+1);
	*/
	if(ehPrimo==NULL){
		printf("erro de alocacao ehPrimo\n");
		exit(1);
	}
	
	for(i = 0; i < tamanhoMemoria; i++)
		ehPrimo[i] = 1;

	for( i = 3; i*i <= para; i+=2 )
	{
		/* pula os multiplos de 3 : 9, 15, 21, 27, ...*/
		if( i >= 3*3 && i % 3 == 0 )
			continue;
		/* pula os multiplos de 5 */
		if( i >= 5*5 && i % 5 == 0 )
			continue;
		/* pula os multiplos de 7 */		
		if( i >= 7*7 && i % 7 == 0 )
			continue;
		/* pula os multiplos de 11 */
		if( i >= 11*11 && i % 11 == 0 )
			continue;
		/* pula os multiplos de 13 */
		if( i >= 13*13 && i % 13 == 0 )
			continue;	

		/* pula numeros antes da parte atual */
		minJ = ((de+i-1)/i)*i;
		if( minJ < i*i )
			minJ = i*i;

		/* valor inicial deve ser impar */
		if( ( minJ & 1) == 0 )
			minJ += i;

		/* encontra todos impares nao-primos */		
		for(j = minJ; j <= para; j += 2 * i){
			indice = j - de;
			ehPrimo[indice/2] = 0;
		}
	}

	/* conta os primos nesse bloco */
	encontrou = 0;
	for( i = 0; i < tamanhoMemoria; i++ )
		encontrou += ehPrimo[i];

	/* dois nao eh impar -> incluir na soma */	
	if( de <= 2 )
		encontrou++;
	
	/*free(ehPrimo);*/

	return 0;
}

/* processa parte por parte, somente numeros primos */
int init_crivo(int sliceSize) {
	
	/*omp_set_num_threads(omp_get_num_procs());*/
	omp_set_num_threads(20);
	
	int found = 0;
	int de = 0, para = 0;
	
	/* cada parte cobre [de..para] */
	#pragma omp parallel for reduction(+:found)
	for( de = 2; de <= MAX_NUM; de += sliceSize ){
		para = de + sliceSize;
		if( para > MAX_NUM )
			para = MAX_NUM;

		found += crivo_singleBlock(de, para);		
	}

	return found;
}

int main(int argc, char* argv[]) {
	int i;

	init_crivo(128*1024); /* Inicializa o crivo de erastotenes */
	
	printf("imprimindo os primos\n");

	for(i = 0; i < 10; i++){
		printf("%d\n", ehPrimo[i]);
	}		

	return 0;
}
