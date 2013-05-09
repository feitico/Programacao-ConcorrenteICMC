#include <stdio.h>
#include <omp.h>

/* Inicializa o crivo de erastotenes */
void init_crivo(){
	/* Declaracao de variaveis */
	int N = 8000; /* o maximo valor no qual procuramos numeros primos */
	int sqrtN = 90; /* a raiz de N */
	int c = 2; /* usado para checar o proximo numero a ser circulado */
	int m = 3; /* usado para chegar o proximo numero a ser marcado */
	int lista[N]; /* a lista de numeros – se lista[x] igual a 1, entao x eh marcado. Senao eh desmarcado. */

	/* passa por todo numero na lista */
	for(c = 2; c <= N-1; c++){
		/* seta todos os numeros como desmarcados */
		lista[c] = 0;
	}

	/* executando ate a raiz de N */
	for(c = 2; c <= sqrtN; c++){
		/* se o numero esta desmarcado */
		if(lista[c] == 0){
			
			/* executa para cada numero maior que c */
			for(m = c+1; m <= N-1; m++){
				/* se m eh multiplo de c*/
				if(m % c == 0){
					/*marcar m*/
					lista[m] = 1;
				}
			}
		}
	}

	/* executa para todo numero na lista */
	for(c = 2; c <= N - 1; c++){
		/* se o numero esta desmarcado */
		if(lista[c] == 0){
			/* o numero e primo */
			printf("%d ", c);
		}
	}
	printf("\n");
	
}

int main(){
	init_crivo();

	return 0;
}
