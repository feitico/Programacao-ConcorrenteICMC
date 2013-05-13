/*compilar make*/
/* executar mpirun -np 2 ./crivo_openmpi -n 10 */
/* executar mpirun -np 1 ./crivo_openmpi -n 1000 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

int main(int argc, char **argv) {
	/* Declaracao de variaveis */
	int N = 16;	/* O numero teto em que procuraremos numeros primos */
	int sqrtN = 0;
	int c = 0;	/* usado para checar o proximo numero a ser circulado */
	int m = 0; /* usado para checar o proximo numero a ser marcado */
	int *lista1; /* A lista de numeros menores que raiz de N */
				/* se lista1[x] igual a 1, entao x e marcado, senao desmarcado */
	int *lista2;	/* A lista de numero maiores que raiz de N */
				/* se lista2[x-L] igual a 1, entao x e marcado, senao desmarcado */
	char next_option = ' '; /* usado para analise de linha de comando */
	int S = 0;	/* Quantidade de numeros que contera a divisao dos numeros 
				   acima da raiz de N pela quantidade de processos*/
	int R = 0;	/* O restante da divisao dos numeros acima da 
				   raiz de N pelo numero de processos */
	int L = 0;	/* O menor numero no split do processo atual */
	int H = 0;	/* O maior numero no split do processo atual */	
	int r = 0; /* A classificacao do processo atual */
	int p = 0; /* O numero total de processos */
	double t1, t2, time_used; /* contabiliza o tempo de inicio e termino do programa */	
	FILE *fileTime; /*Ponteiro do arquivo de saída*/	

	/* Inicializa o ambiente mpi */
	MPI_Init(&argc, &argv);

	//	
	t1 = MPI_Wtime();

	/* Determina a classificacao do processo atual e 
		o numero de processos */
	MPI_Comm_rank(MPI_COMM_WORLD, &r);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* analise dos argumentos da linha de comando */
    while((next_option = getopt(argc, argv, "n:")) != -1) {
        switch(next_option) {
            case 'n':
                N = atoi(optarg);
                break;
            case '?':
            default:
                fprintf(stderr, "Usage: %s [-n N]\n", argv[0]);
                exit(-1);
        }
    }

	/* Calcula a raiz de N */
	sqrtN = (int)sqrt(N);

	/* Calculando S, R, L e H */
	S = (N-(sqrtN+1)) / p;
	R = (N-(sqrtN+1)) % p;
	L = sqrtN + r*S + 1;
	H = L+S-1;
	
	if(r == p-1)
		H += R;	

	/* Alocando memoria para as listas */
	lista1 = (int*)malloc((sqrtN+1) * sizeof(int));
	lista2 = (int*)malloc((H-L+1) * sizeof(int));

	/* se da erro no malloc */
    if(lista1 == NULL || lista2 == NULL) {
        fprintf(stderr, "Sorry, there was an internal error. Please run again.\n");
        exit(-1);
    }

	/* Roda por todos os numeros na lista1 */
	for(c = 2; c <= sqrtN; c++){
		/* seta todos os numeros como desmarcado */
		lista1[c] = 0;
	}	

	/* Roda por todos os numeros na lista2 */
	for(c = L; c <= H; c++){
		/* seta todos os numeros como desmarcado */
		lista2[c-L] = 0;
	}
	
	/* Passa por todos os numeros na lista1 */
	for(c = 2; c <= sqrtN; c++){
		/* se o numero esta desmarcado */
		if(lista1[c] == 0){
			/* Passa por todos os numeros maiores que c na lista1 */
			for( m = c+1; m <= sqrtN; m++ ){
				/* se m eh multiplo de c */
				if( m % c == 0 ){
					/* marcar m */
					lista1[m] = 1;
				}

			}

			/* Passa por todos os numeros maiores que c na lista2 */
			for( m = L; m <= H; m++ ){
				/* se m eh multiplo de c */
				if( m % c == 0 ){
					/* marcar m */
					lista2[m-L] = 1;
				}
			}
		}
	}

	/* se a classificacao do processo atual eh 0 */
	if( r == 0 ){
		/* passa por todos os numeros na lista 1 */
		/*		
		for( c = 2; c <= sqrtN; c++ ){
			// se o numero esta desmarcado
			if(lista1[c] == 0){
				// o numero eh primo, imprimir
				printf("%1u ", c);
			}
		}
		*/ 

		/* passa por todos os valores na lista 2 */
		/*
		for( c = L; c <= H; c++ ){
			// se o numero esta desmarcado
			if(lista2[c-L] == 0){
				// o numero eh primo, imprimir
				printf("%1u ", c);			
			}
		}
		*/
		
		/* passa por todos os processos */
		for( r = 1; r <= p-1; r++ ){
			/* Calcula L e H por r */	
			L = sqrtN + r*S + 1;
			H = L+S-1;
			if ( r == p-1 )
				H += R;

			/* Recebe a lista 2 de processos */
			MPI_Recv(lista2, H-L+1, MPI_INT, r, 0, 
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			/* passa pela lista2 que recebemos */
			/*			
			for(c = L; c <= H; c++){
				// se o numero esta desmarcado
				if(lista2[c-L] == 0){
					// o numero eh primo, imprimir
					printf("%1u ", c);
				}
			}
			*/						
		}
		/*printf("\n");*/

		/* se o processo nao eh classificado como 0 */
	}else{
		/* envia a lista2 para o processo 0 */
		MPI_Send(lista2, H-L+1, MPI_INT, 0, 0, 
				 MPI_COMM_WORLD);
	}
	
	/* desalocando memoria das listas */
	free(lista2);
	free(lista1);
	
	t2 = MPI_Wtime();
	
	time_used = t2 - t1;

	/* Finaliza o ambiente MPI */
	MPI_Finalize();
	
	fileTime = fopen("tempo_crivo_milhao.txt", "a");
	fprintf(fileTime, "%d, %f\n", p, time_used);
	fclose(fileTime);		

	return 0;
}
