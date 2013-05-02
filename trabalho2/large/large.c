#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_NUM 8000
#define RAIZ_MAX_NUM 90

int primo[MAX_NUM];

/* Inicializa o crivo de erastotenes */
void init_crivo() {
	int i, j;
	for(i=2; i<MAX_NUM;i++)
		primo[i] = i;
	
	for(i=2; i<=RAIZ_MAX_NUM; i++) {
		if(primo[i] == i) {
			for(j=i+i; j<MAX_NUM; j+=i)
				primo[j] = 0;
		}
	}
}

/* Retorna 1 se for primo */
int isPrimo(int num) {
	return primo[num];
}

/* Retorna 1 se for palindromo */
int palindromo(char* str) {
	int i;
	int half;
	int length = strlen(str);

	if(length == 1)
		return 0;

	if(length % 2 == 0)
		half = length / 2;
	else
		half = (length / 2) + 1;


	for(i=0; i<=half; i++) {
		if(str[i] != str[length-i-1]) {
			return 0;
		}
	}

	return 1;
}

int main() {
	FILE* pWikipedia;
	char str[1024];
	int num, i, max=0;

	pWikipedia = fopen("wikipedia.txt", "r");
	init_crivo(); /* Inicializa o crivo de erastotenes */

	while(!feof(pWikipedia)) {
		if(fscanf(pWikipedia, "%s", str) != EOF) {
			palindromo(str);
			/*	
			if(palindromo(str)) {
				num = 0;
				for(i = 0; i<strlen(str); i++)
					num += (int) str[i];
				if(isPrimo(num) != 0)
					printf("%s = %d eh primo!\n", str, num);

				if(max < num) {
					max = num;
				}
			}*/
		} 
	}
/*
	printf("max: %d\n", max);
*/	
	fclose(pWikipedia);

	return 0;
}
