#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>
#include <iostream>
#include <set>

using namespace std;

//Marca o tempo de inicio
void tic();
//Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
double toc();

// Read the whole file
char* words;

struct ltstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};

map<int, set<char*, ltstr> > dict_by_length;
map<int, set<char*, ltstr> > founded_by_length;

char* palavra_random();

int main(int argc, char* argv[]) {
	size_t total;
	double elapsed_time;
	FILE *arq_palavras;
	int i, begin_word;
	char* temp;
	char* pch;
	int dict_size;

	tic();
	arq_palavras = fopen(argv[1], "r");

	fseek(arq_palavras, 0L, SEEK_END);
	int sz = ftell(arq_palavras);

	fseek(arq_palavras, 0L, SEEK_SET);	

	words = (char*) malloc (sizeof(char) * sz);
	total = fread(words, 1, sz, arq_palavras);
	
	// Parsea as palavras lidas
	pch = strtok (words, " ,\n");

	//for(int i=0; i<100; i++) {
	while(pch != NULL){
		//printf ("%s\n",pch);
		int length = strlen(pch);
		dict_by_length[length].insert(pch);
		
		//dict.insert(pch);
		pch = strtok (NULL, " ,\n");
		//getchar();

	}

	int qtd_five_first=0;
	for(int i=1; i<=5; i++)
		qtd_five_first += dict_by_length[i].size();
	cout << qtd_five_first << endl;
	
	int v[6];
	for(int i=1; i<=5; i++) { 
		v[i] = 0;
		cout << i << " prop: " << dict_by_length[i].size() / (float)qtd_five_first << endl;
	}
/*
	map<int, set<char*,ltstr> >::iterator it1;
	for(it1=dict_by_length.begin(); it1!=dict_by_length.end(); ++it1) {
		cout << it1->first << " : " << it1->second.size() << endl;
	}
*/
while(true) {
	for(int i=1; i<=5; i++)
		v[i] = 0;

	for(int i=0; i<qtd_five_first; i++) {
		int r = rand() % qtd_five_first;
		if(r < dict_by_length[1].size())
			v[1]++;
		else if(r < dict_by_length[2].size())
			v[2]++;
		else if(r < dict_by_length[3].size())
			v[3]++;
		else if(r < dict_by_length[4].size())
			v[4]++;
		else
			v[5]++;
	}
	int diff=0;
	for(int i=1; i<=5; i++) { 
		diff += abs(v[i] - (int)dict_by_length[i].size());
		cout << i << " rand: " << v[i] << " real: " << dict_by_length[i].size() << " diff: " << v[i] - (int)dict_by_length[i].size() << endl;
	}
	cout << "total diff: " << diff << endl;

	getchar();
	getchar();
	}
/*
	int k=0;
	while(true) {
		char * new_word = palavra_random();
		it = dict.find(new_word);

		if(it != dict.end()) {
			cout << "encontrou: " << new_word << endl;
			achadas.insert(*it);
			dict.erase(it);
			cout << achadas.size() << endl;
			k++;
			if(k == 2)
				break;
		} else {
			cout << "nao: " << new_word << endl;
		}
	}	
*/
/*
	for(it = dict.begin(); it != dict.end(); ++it){
		cout << "a: " << *it << endl;
	}
*//*		
	for(it = achadas.begin(); it != achadas.end(); ++it){
		cout << "achei: " << *it << endl;
	}
*/
	// Parsea as palavras lidas
	/*
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
*/
	fclose(arq_palavras);
	elapsed_time = toc();
	printf("elapsed time: %f\ntotal %d\n", elapsed_time, (int)total);

	free(words);

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

char* palavra_random() {
	char* str;
	int tamanhoPalavra;

	tamanhoPalavra = rand() % 5 + 1;
	str = (char*) malloc(sizeof(char) * (tamanhoPalavra+1));
	
	for(int i = 0; i < tamanhoPalavra; i++){
		str[i] = ('a' + rand() % 26);
	}
	str[tamanhoPalavra] = '\0';
	
	return str;
}
