#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>
#include <iostream>
#include <set>
#include "omp.h"

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
char** block; // Le n blocos do arquivo

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int total_palavras; //Total de palavras a serem encontradas
int qtd_encontradas=0; //Total de palavras encontradas

clock_t begin; //Utilizado pelas funcoes tic() e toc() para calcular marcar o tempo inicial

typedef set<char*, strcompare> Dict;
typedef map<int, Dict > MapDict; // Representa um mapa de dicionarios, sendo a chave o numero de letras das palavras do dicionario
MapDict dict_by_length; //Dicionario com as palavras ainda nao encontradas
MapDict founded_by_length; //Dicionario com as palavras ja encontradas

int n_threads;

int main(int argc, char* argv[]) {

	if(argc != 3) {
		cout << "Uso: " << argv[0] << " palavras.txt n_threads" << endl;
		exit(-1);
	}
	n_threads = atoi(argv[2]);
	omp_set_num_threads(n_threads);
	//Inicializa o dicionario
	init_dict(argv[1]);	
	
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
	int size = ftell(arq_palavras) / n_threads;
	fseek(arq_palavras, 0L, SEEK_SET);

	block = (char**) malloc(sizeof(char*) * n_threads);

	for(int i=0; i<n_threads; i++) {
		block[i] = (char*) malloc (sizeof(char) * size);
		fread(block[i], 1, size, arq_palavras);	
	}
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
	int my_rank, j, length;
	char* pch;
	char* save_ptr;
/*
	char** block = (char**) malloc(sizeof(char*) * n_threads);
	for(i=0; i<n_threads; i++) {
		block[i] = (char*) malloc((sizeof(text) / 4) + 1);
		memcpy(block[i], text + i*(sizeof(text) / 4), sizeof(text) / 4);
	}
*/

	#pragma omp parallel num_threads(n_threads) \
		private(my_rank, j, pch, save_ptr, length) shared(block, dict_by_length)
	{
		my_rank = omp_get_thread_num();
		//printf("Thread %d\n", my_rank);
		j = 0;
		pch = strtok_r(block[my_rank], " ,\n", &save_ptr);                                    
    	//for(int i=0; i<100; i++) {
   		while(pch != NULL){
    		length = strlen(pch);
			//printf("%d - token %d : %s\n", my_rank,j,pch);
			dict_by_length[length].insert(strdup(pch)); //Insere uma copia no dicionario das palavras de tamanho i    
    		pch = strtok_r (NULL, " ,\n", &save_ptr);                                   
			j++;
    	}                                                                 
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
