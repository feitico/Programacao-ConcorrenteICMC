#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
	FILE* pWikipedia;
	char str[1024];
	int num, i, max=0;

	pWikipedia = fopen("wikipedia.txt", "r");

	while(!feof(pWikipedia)) {
		if(fscanf(pWikipedia, "%s", str) != EOF) {
			if(palindromo(str)) {
				num = 0;
				for(i = 0; i<strlen(str); i++)
					num += (int) str[i];
				if(max < num) {
					max = num;
				}
			}
		} 
	}

	/*printf("max: %d\n", max);*/
	
	fclose(pWikipedia);

	return 0;
}
