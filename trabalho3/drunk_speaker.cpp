#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>
#include <iostream>
#include <set>

using namespace std;

/* Funcoes auxiliares */
void tic(); //Marca o tempo de inicio
double toc();//Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
void ler_arquivo();// Le todo o arquivo
void calc_proporcao();// Calcula a proporcao de palavras com ateh 5 letras
void calc_total_palavras();//Calcula total de palavras
void init_dict(const char* filename);// Inicializa o dicionario
int gera_tamanho_palavra();// Gera o tamanho de uma palavra de forma uniforme
char* gera_palavra(int tamanho);// Gera uma palavra com um certo tamanho
void imprime_prop(); //Imprime a proporcao de palavras encontradas
void parser();// Parsea as palavras lidas
char* mystrcat(char* str1, char* str2);// Aloca e concatena duas strings

/* 
	Estrutura auxiliar para comparar ponteiros de char na estrutura set
*/
struct strcompare
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};

/* Variaveis globais */
FILE *arq_palavras; //Ponteiro para o arquivo
char* text; // Armazena todo o texto do arquivo

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int total_palavras; //Total de palavras a serem encontradas
int qtd_encontradas=0; //Total de palavras encontradas

clock_t begin; //Utilizado pelas funcoes tic() e toc() para calcular marcar o tempo inicial

typedef set<char*, strcompare> Dict;
typedef map<int, Dict > MapDict; // Representa um mapa de dicionarios, sendo a chave o numero de letras das palavras do dicionario
MapDict dict_by_length; //Dicionario com as palavras ainda nao encontradas
MapDict founded_by_length; //Dicionario com as palavras ja encontradas

int main(int argc, char* argv[]) {

	if(argc != 2) {
		cout << "Uso: " << argv[0] << " palavras.txt" << endl;
		exit(-1);
	}
	//Inicializa o dicionario
	init_dict(argv[1]);	
	
	MapDict::iterator it1;
	for(it1=dict_by_length.begin(); it1!=dict_by_length.end(); ++it1) {
		cout << it1->first << " : " << it1->second.size() << endl;
	}
	
	cout << "map size: " << dict_by_length.size() << endl;

	Dict::iterator itset;
	tic();
	while(true) {
		int tamanho = gera_tamanho_palavra();
		//int tamanho = rand() % 5 + 1;
		char *new_word = gera_palavra(tamanho);

		itset = dict_by_length[tamanho].find(new_word);

		free(new_word);

		if(itset != dict_by_length[tamanho].end()) {
			founded_by_length[tamanho].insert(*itset);
			dict_by_length[tamanho].erase(itset);
			qtd_encontradas++;
			
			imprime_prop();

			float prop = qtd_encontradas / (float) total_palavras;
			if(prop >= 0.5)
				break;
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
/*	
	int nindices = dict_by_length.size();
	int* indices = (int*) calloc(sizeof(int) * nindices);

	// Copia os tamanhos de palavras disponiveis
	vector<int> tamanhos;
	tamanhos.push_back(0);
	for(MapDict::iterator it=dict_by_length.begin(); it!=dict_by_length.end(); ++it)
		tamanhos.push_back(it->first);

	//Concatena as palavras ja encontradas
	int k = 0;
	while(1) {
		
	}


*/
	char* combined[2];
	int k=0;
	//combined = (char**) malloc(sizeof(char*) * 2);

	// Inicia todos os iteradores para o primero número de palavras
	MapDict::iterator it[2];
	for(int i=0; i<2; i++)
		it[i] = founded_by_length.begin();

	while(1) {
		Dict::iterator it_word[2];

		while(1) {
			for(int i=0; i<2; i++)
				it_word[i] = it[i]->second.begin();

			int tamanho = it[0]->first + it[1]->first;
			combined[0] = mystrcat(*it_word[0], *it_word[1]);
			combined[1] = mystrcat(*it_word[1], *it_word[0]);

			for(int i=0; i<2; i++) {
				if(dict_by_length[tamanho].find(combined[i]) != dict_by_length[tamanho].end()) { //Verifica se a palavra combinada existe
					cout << qtd_encontradas << "," << k << " " << tamanho << ": " << combined[i] << endl;
					qtd_encontradas++;
					k++;
					imprime_prop();
				}
			}

			free(combined[0]);
			free(combined[1]);
		}
	}
/*
	int k=0;
	for(MapDict::iterator founded_it1=founded_by_length.begin(); 
	    founded_it1!=founded_by_length.end(); 
            ++founded_it1) {
		for(MapDict::iterator founded_it2=founded_by_length.begin(); 
		    founded_it2!=founded_by_length.end(); 
		    ++founded_it1) {
			for(Dict::iterator palavra_it1=founded_it1->second.begin(); 
			    palavra_it1!=founded_it1->second.end(); 
			    ++palavra_it1) {
				for(Dict::iterator palavra_it2=founded_it2->second.begin(); 
				    palavra_it2!=founded_it2->second.end(); 
				    ++palavra_it2) {
					int tamanho = founded_it1->first + founded_it2->first;
					char* combined1 = mystrcat(*palavra_it1, *palavra_it2);
					char* combined2 = mystrcat(*palavra_it2, *palavra_it1);

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
	*/
	fclose(arq_palavras);
	cout << "elapsed time: " << toc() << endl;

	return 0;
}

// Aloca e concatena duas strings
char* mystrcat(char* str1, char* str2) {
	char* cat = (char*) malloc(sizeof(char) * (strlen(str1) + strlen(str2) + 1));
	strcpy(cat, str1);
	strcat(cat, str2);
	return cat;
}

// Marca o tempo de inicio
void tic() {
	begin = clock();
}

// Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
double toc() {
	clock_t end = clock();
	return double(end - begin) / CLOCKS_PER_SEC;
}

// Le todo o arquivo
void ler_arquivo(const char* filename) {
	arq_palavras = fopen(filename, "r");

	//Calcula o tamanho do arquivo
	fseek(arq_palavras, 0L, SEEK_END);
	int size = ftell(arq_palavras);
	fseek(arq_palavras, 0L, SEEK_SET);

	text = (char*) malloc (sizeof(char) * size);
	fread(text, 1, size, arq_palavras);	
}

// Calcula a proporcao de palavras com ateh 5 letras
void calc_proporcao() {
	qtd_palavra[0] = 0;
	for(int i=1; i<=5; i++) {	
		qtd_palavra[i] = dict_by_length[i].size();
		qtd_palavra[0] += qtd_palavra[i];
	}
	cout << "total prop: " << qtd_palavra[0] << endl;
	total_palavras = qtd_palavra[0]; /* Change it after */
}

//Calcula total de palavras
void calc_total_palavras() {
    total_palavras = 0;
    map<int, set<char*,strcompare> >::iterator it;
    for(it = dict_by_length.begin(); it!=dict_by_length.end(); ++it)
    	total_palavras += it->second.size();
                                                                     
    cout << "total de palavras: " << total_palavras << endl;
}

// Inicializa o dicionario
void init_dict(const char* filename) {
	tic();
	ler_arquivo(filename);
	cout << "leitura: " << toc() << endl;
	
	tic();
	parser();
	cout << "parser: " << toc() << endl;

	calc_proporcao();

	//calc_total_palavras(); Change it after
}

// Parsea as palavras lidas
void parser() {
    char *pch = strtok (text, " ,\n");
                                            
    //for(int i=0; i<100; i++) {
    while(pch != NULL){
    	int length = strlen(pch);
    	dict_by_length[length].insert(strdup(pch)); //Insere uma copia no dicionario das palavras de tamanho i    
    	pch = strtok (NULL, " ,\n");                                    
    }
                                                                         
    free(text);
}

// Gera o tamanho de uma palavra de modo uniforme
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

// Gera uma palavra com um certo tamanho
char* gera_palavra(int tamanho) {
	char* str;

	str = (char*) malloc(sizeof(char) * (tamanho));
	
	for(int i = 0; i < tamanho; i++){
		str[i] = ('a' + rand() % 26);
	}
	str[tamanho] = '\0';
	
	return str;
}

//Imprime a proporcao de palavras encontradas
void imprime_prop() {
	static int controle=1; // Controle deve ser preservado

	float prop = qtd_encontradas / (float) total_palavras;
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
