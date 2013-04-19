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
// Retorna 1 se for palindromo
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
	FILE* pShakespeare;
	char str[1024];
	int num, i, max=0;

	pShakespeare = fopen("shakespeare.txt", "r");

	while(!feof(pShakespeare)) {
		if(fgets(str, 1024, pShakespeare) != NULL) {
			whitespace(str);
			if(palindromo(str)) {
				printf("str: %s\n", str);
				num = 0;
				for(i = 0; i<strlen(str); i++)
					num += (int) str[i];
				if(max < num) {
					max = num;
					printf("max: %d\n", max);
				}
			}
		} 
	}

	printf("max: %d\n", max);
	
	fclose(pShakespeare);

	return 0;
}
