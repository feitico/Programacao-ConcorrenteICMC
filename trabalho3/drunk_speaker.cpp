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

//Representa a qtd de cada palavras no dicionario de tamanho 1,2,3,4 e 5
int qtd_palavra[6];

FILE *arq_palavras;

void ler_arquivo() {
	fseek(arq_palavras, 0L, SEEK_END);
	int size = ftell(arq_palavras);
	fseek(arq_palavras, 0L, SEEK_SET);

	words = (char*) malloc (sizeof(char) * size);
	fread(words, 1, size, arq_palavras);	
}

void calc_proporcao() {
	qtd_palavra[0] = 0;
	for(int i=1; i<=5; i++) {	
		qtd_palavra[i] = dict_by_length[i].size();
		qtd_palavra[0] += qtd_palavra[i];
	}
}

void init_dict(const char* filename) {
	arq_palavras = fopen(filename, "r");
	
	//Le o arquivo
	ler_arquivo();

	// Parsea as palavras lidas
	char *pch = strtok (words, " ,\n");
                                            
	//for(int i=0; i<100; i++) {
	while(pch != NULL){
		//printf ("%s\n",pch);
		int length = strlen(pch);
		dict_by_length[length].insert(pch);
	
		pch = strtok (NULL, " ,\n");                                    
	}

	calc_proporcao();
}

int gera_tamanho_palavra() {
	int r = rand() % qtd_palavra[0];
        if(r < qtd_palavra[1])
        	return 1;
        else if(r < qtd_palavra[2])
        	return 2;
        else if(r < qtd_palavra[3])
        	return 3;
        else if(r < qtd_palavra[4])
        	return 4;
        else
        	return 5;
}

char* gera_palavra(int tamanho) {
	char* str;

	str = (char*) malloc(sizeof(char) * (tamanho));
	
	for(int i = 0; i < tamanho; i++){
		str[i] = ('a' + rand() % 26);
	}
	str[tamanho] = '\0';
	
	return str;
}

int main(int argc, char* argv[]) {

	//Inicializa o dicionario
	init_dict(argv[1]);

	map<int, set<char*,ltstr> >::iterator it1;
	for(it1=dict_by_length.begin(); it1!=dict_by_length.end(); ++it1) {
		cout << it1->first << " : " << it1->second.size() << endl;
	}

	int k=0;
	set<char*, ltstr>::iterator itset;
	int controle=1;
	tic();
	while(true) {
		int tamanho = gera_tamanho_palavra();
		//int tamanho = rand() % 5 + 1;
		char *new_word = gera_palavra(tamanho);

		itset = dict_by_length[tamanho].find(new_word);

		free(new_word);
		if(itset != dict_by_length[tamanho].end()) {
			float prop = k / (float) qtd_palavra[0];
			//cout << prop * 100 << "% " <<  k << " encontrou: " << new_word << endl;
			founded_by_length[tamanho].insert(*itset);
			dict_by_length[tamanho].erase(itset);
			k++;

			if((prop >= 0.1) && (controle == 1)) {
				cout << "10% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.2) && (controle == 2)) {
				cout << "20% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.3) && (controle == 3)) {
				cout << "30% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.4) && (controle == 4)) {
				cout << "40% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.5) && (controle == 5)) {
				cout << "50% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.6) && (controle == 6)) {
				cout << "60% encontrado: " << toc() << endl;
				controle++;	
				break;
			} else if ((prop >= 0.7) && (controle == 7)) {
				cout << "70% encontrado: " << toc() << endl;
				controle++;	
			} else if ((prop >= 0.8) && (controle == 8)) {
				cout << "80% encontrado: " << toc() << endl;
				controle++;
			} else if ((prop >= 0.9) && (controle == 9)) {
				cout << "90% encontrado: " << toc() << endl;
				controle++;
			}	
		}
	}

	/*
		Junta palavras de até 5 letras para gerar palavras maiores

		found_it1 = indice do dicionario de palavras de tamanho i
		found_it2 = indice do dicionario de palavras de tamanho j
		
		palavra_it1 = indice da palavra k do dicionario i
		palavra_it2 = indice da palavra l do dicionario j

		tam_palavras = 1,2,3,4,5, ... 45

		1 + 1 esta contido tam_palavrras 
		thread <- (1,1) -> encontrou
		thread <- (1,2)
		thread <- (1,45) - X

		thread <- (45, 45)

		thread <- (1,1,1)

		thread <- (_,_,_,0,_,_,_,_,_,_,_) - 45 tamanho	
	*/	
	k = 0;


	map<int, set<char*, ltstr> >::iterator founded_it1, founded_it2;
	set<char*, ltstr>::iterator palavra_it1, palavra_it2;
	for(founded_it1=founded_by_length.begin(); 
	    founded_it1!=founded_by_length.end(); 
            ++founded_it1) {
		for(founded_it2=founded_by_length.begin(); 
		    founded_it2!=founded_by_length.end(); 
		    ++founded_it1) {
			for(palavra_it1=founded_it1->second.begin(); 
			    palavra_it1!=founded_it1->second.end(); 
			    ++palavra_it1) {
				for(palavra_it2=founded_it2->second.begin(); 
				    palavra_it2!=founded_it2->second.end(); 
				    ++palavra_it2) {
					int tamanho = founded_it1->first + founded_it2->first;
					char* combined1 = (char*) malloc(sizeof(char) * tamanho);
					char* combined2 = (char*) malloc(sizeof(char) * tamanho);
				
					strcpy(combined1, *palavra_it1);
					strcat(combined1, *palavra_it2);

					strcpy(combined2, *palavra_it2);
					strcat(combined2, *palavra_it1);

					itset = dict_by_length[tamanho].find(combined1);
					if(itset != dict_by_length[tamanho].end()) {
						cout << k << " " << tamanho << " concatenei: " << *palavra_it1 << "+" << *palavra_it2 << endl;
						k++;
					}			
					itset = dict_by_length[tamanho].find(combined2);
					if(itset != dict_by_length[tamanho].end()) {
						cout << k << " " << tamanho << " concatenei: " << *palavra_it2 << "+" << *palavra_it1 << endl;
						k++;
					}			
					free(combined1);
					free(combined2);
				}
			}
		}
	}
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
	cout << "elapsed time: " << toc() << endl;

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















