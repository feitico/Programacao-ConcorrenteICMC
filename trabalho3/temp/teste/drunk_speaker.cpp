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
int controle=1;

typedef set<char*, strcompare> Dict;
typedef map<int, Dict > MapDict; // Representa um mapa de dicionarios, sendo a chave o numero de letras das palavras do dicionario
Dict mydict;
Dict mydict_found;
MapDict dict_by_length; //Dicionario com as palavras ainda nao encontradas
MapDict founded_by_length; //Dicionario com as palavras ja encontradas

int main(int argc, char* argv[]) {

	if(argc != 2) {
		cout << "Uso: " << argv[0] << " palavras.txt" << endl;
		exit(-1);
	}
	//Inicializa o dicionario
	init_dict(argv[1]);	

	//Gera as palavras com até cinco letras
	Dict::iterator itset;
	tic();
	while(true) {
		int tamanho = gera_tamanho_palavra();
		char *new_word = gera_palavra(tamanho);

		itset = mydict.find(new_word);

		if(itset != mydict.end()) {
			mydict_found.insert(*itset);
			mydict.erase(*itset);
			qtd_encontradas++;
			
			imprime_prop();

			float prop = qtd_encontradas / (float) total_palavras;
			if(prop >= 1.0)
				break;
		}
		
		free(new_word);
	}

	cout << mydict_found.size() << ":" << mydict.size() << endl;

	total_palavras = mydict.size();

	controle=1;

	// Junta duas palavras para gerar palavras maiores
	Dict::iterator it[2];
	char* combined[2];
	for(it[0] = mydict_found.begin(); it[0]!=mydict_found.end(); ++it[0]) {
		for(it[1] = mydict_found.begin(); it[1]!=mydict_found.end(); ++it[1]) {
			combined[0] = mystrcat(*it[0], *it[1]);
			combined[1] = mystrcat(*it[1], *it[0]);
			
			for(int i=0; i<2; i++) {
				itset = mydict.find(combined[i]);
				if(itset != mydict.end()) {
					//cout << qtd_encontradas << " - combined: " << combined[i] << endl;
					mydict_found.insert(*itset);
					mydict.erase(*itset);
					qtd_encontradas++;

					imprime_prop();

					float prop = qtd_encontradas / (float) total_palavras;
					if(prop >= 1.0)
						break;
				}
				free(combined[i]);
			}
		}
	}
		
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
	for(int i=0; i<6; i++)
    	qtd_palavra[i] = 0;
    
    Dict::iterator it;
    for(it=mydict.begin(); it!=mydict.end(); ++it) {
    	if(strlen(*it) <=5 ) {
    		qtd_palavra[strlen(*it)]++;
    		qtd_palavra[0]++;
    	}
    }
	
    for(int i=0; i<6; i++)
        cout << i << ":" << qtd_palavra[i] << endl;

	cout << "total prop: " << qtd_palavra[0] << endl;
    total_palavras = qtd_palavra[0];
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

	//total_palavras = mydict.size();
}

// Parsea as palavras lidas
void parser() {
    char *pch = strtok (text, "\n");
                                            
    while(pch != NULL){
	
		mydict.insert(strdup(pch));
    	pch = strtok (NULL, "\n");                                    
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
    } else if (prop == 1.0) {
		cout << "100% encontrado: " << toc() << endl;
	}
}
