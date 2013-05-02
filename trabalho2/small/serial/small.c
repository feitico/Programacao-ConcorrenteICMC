#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

void whitespace(char* str) {
	char *i=str;
	char *j=str;
	while(*j != '\n') {
		*i = *j++;
		if( !isspace(*i) )
			i++;
	}
	*i = 0;
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

int isendpoint(char c) {
	switch(c) {
		case '.':
		case '!':
		case '?':
		case '\n':
			return 1;
		default:
			return 0;
	}
}

int main(int argc, char* argv[]) {
	char str[1024], c;
	int num, i, max=0;
	FILE* entrada;
	char* frase;

	if(argc != 2) {
		printf("Usage: ./small shakespeare.txt\n");
		exit(-1);
	}

	entrada = fopen(argv[1], "r");


	while(!feof(entrada)) {
		/*
		i = 0;
		str[i] = '\0';
		while(1) {
			if( (c = fgetc(entrada)) != EOF) {
				if(isendpoint(c)) {
					str[i] = '\0';
					break;
				} else {
					if(!isspace(c)) {
						str[i++] = c;	
					}
				}
			}
		}

		printf("frase: %s\n", str);
		*/
		
		if(fgets(str, 1024, entrada) != NULL) {
			frase = strtok(str, ".!?\n");

			while(frase) {
				printf("frase: %s-", frase);
				whitespace(frase);
				printf("%s\n", frase);
				frase = strtok(NULL, ".!?\n");
			}/*
			whitespace(str);
			if(palindromo(str)) {
				printf("str: %s\n", str);
				num = 0;
				for(i = 0; i<strlen(str); i++)
					num += (int) str[i];
				if(max < num) {
					max = num;
				}
			}*/
		} 
	}
	free(str);

	/*printf("max: %d\n", max);*/
	
	fclose(entrada);

	return 0;
}
