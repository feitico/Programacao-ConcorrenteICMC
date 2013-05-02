#include <stdio.h>
#include <stdlib.h>

/* 
 * 1 argumento - numero de particoes de arquivo
 * 2 argumento - arquivo texto a ser particionado
 */
int main(int argc, char *argv[]) {
	FILE* entrada=fopen(argv[1], "r");
	char str[1024];

	/* Percorro o arquivo de entrada e particiono no arquivo de saida */
	while(!feof(entrada)) {
		fscanf(entrada, "%s", str);
	}

	/* Fecha os arquivos */
	fclose(entrada);

	return 0;
}

