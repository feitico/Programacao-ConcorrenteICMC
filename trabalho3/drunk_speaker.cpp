#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>

#define FILE_SIZE 24823019
using namespace std;

//Marca o tempo de inicio
void tic();
//Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
double toc();

// Read the whole file
char words[FILE_SIZE];

map<string, int> dict;

int main(int argc, char* argv[]) {
	size_t total;
	double elapsed_time;
	FILE *arq_palavras;
	int i, begin_word;
	char* temp;
	
	tic();
	arq_palavras = fopen(argv[1], "r");

	total = fread(words, 1, FILE_SIZE, arq_palavras);
	
	// Parsea as palavras lidas
	begin_word = 0;
	for(i=0; i<FILE_SIZE; i++) {
		if( (words[i] == ',') ||
			(words[i] == ' ') ||
			(words[i] == '\n') ) {
			temp = (char*) malloc(sizeof(char) * (i - begin_word));
			strncpy(temp, words + begin_word, i - begin_word);
			begin_word = i+1;

			dict[string(temp)] = 0; // Indica que a palavra existe, mas ainda nao foi encontrado
			//free(temp);
		}
	}

	map<string, int>::iterator it;
	for(it=dict.begin(); it!=dict.end(); ++it) {
		printf("%s %d\n", it->first.c_str(), it->second);
	}

	fclose(arq_palavras);
	elapsed_time = toc();
	printf("elapsed time: %f\ntotal %d\n", elapsed_time, (int)total);

	return 0;
}

clock_t begin=0;

void tic() {
	begin = clock();
}

double toc() {
	clock_t end = clock();
	return double(end - begin) / CLOCKS_PER_SEC;
}
