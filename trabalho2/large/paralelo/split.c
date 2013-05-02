#include <stdio.h>
#include <stdlib.h>

/* 
 * 1 argumento - numero de particoes de arquivo
 * 2 argumento - arquivo texto a ser particionado
 */
int main(int argc, char *argv[]) {
	int n = atoi(argv[1]);
	FILE* entrada=fopen(argv[2], "r");
	FILE* saida[n];
	int i;
	char buffer[20];
	char str[1024];

	/* Cria os arquivos particionados */
	for(i=0; i<n; i++) {
		sprintf(buffer, "split%d.txt", i);
		saida[i] = fopen(buffer, "w");
	}
	
	i=0;
	/* Percorro o arquivo de entrada e particiono no arquivo de saida */
	while(!feof(entrada)) {
		if(fscanf(entrada, "%s", str) != EOF) {
			fprintf(saida[i], "%s ", str);
			i = (i+1) % n;
		}		
	}

	/* Fecha os arquivos */
	fclose(entrada);
	
	for(i=0; i<n; i++) {
		fclose(saida[i]);
	}

	return 0;
}

