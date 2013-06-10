#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>
#include <iostream>
#include <set>
#include <sys/time.h>
#include "omp.h"
#include <mpi.h>

#include "utils.h"
#include "dict.hpp"

#define TAG_SHORT 7
#define GENERATE 100000

using namespace std;

Dict shortDict;
Dict compoundDict;

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int totalPalavras; //Total de palavras a serem encontradas
int total_prop;
//int qtd_encontradas=0; //Total de palavras encontradas

char* text;
int filesize; // tamanho do arquivo

/* Funcoes auxiliares */
void readDictFile();// Le todo o arquivo
void calc_proporcao();// Calcula a proporcao de palavras com ateh 5 letras
void calc_totalPalavras();//Calcula total de palavras

void loadDict(const char* filename);// Inicializa o dicionario


int gera_tamanho_palavra(struct drand48_data &buffer);// Gera o tamanho de uma palavra de forma uniforme
char* gera_palavra(int tamanho,struct drand48_data &buffer);// Gera uma palavra com um certo tamanho
//void imprime_prop(); //Imprime a proporcao de palavras encontradas
void parser();// Parsea as palavras lidas
char* mystrcat(const char* str1,const char* str2);
int entryCompare( const void * a, const void * b); //Compara duas entrys
int dict_mark(const char* word); // Marca uma palavra no dicionario
void seed_rand(int thread_n, struct drand48_data *buffer);
char *mysubstr(char *str, int begin, int end);
double myrand(struct drand48_data &buffer);
void readText(const char* filename); //Le o arquivo inteiro
void initDict(); // Inicializa o dicionario
void generateWords(int id, int qtd, int debug); //Gera palavras com ate 5 letras de forma aleatoria
void imprime_prop(int &controle, int qtdMarked, int total);

int main(int argc, char* argv[]) {
    MPI::Status mpi_status;
	double start, end;
    int porcent;
    int chunkSize;

    if(argc != 4) {
        cout << "Uso: " << argv[0] << " palavras.txt porcentagem chunkSize" << endl;
        exit(-1);
    }
    porcent = atoi(argv[2]);
    chunkSize = atoi(argv[3]);

    MPI::Init(argc, argv);

    int id = MPI::COMM_WORLD.Get_rank();
    //int numprocs = MPI::COMM_WORLD.Get_size();

	MPI::COMM_WORLD.Barrier();
	start = MPI::Wtime();
    
    if(id == 0) {
	    readText(argv[1]); //Somente o master le o arquivo inteiro

    	end = MPI::Wtime();

		cout << "read file time: " << end - start << endl;
    }

    //Envia o tamanho do buffer para todos os processadores
    MPI::COMM_WORLD.Bcast(&filesize, 1, MPI::INT, 0); // qtd de palavras

    // Aloca o text para os outros processos
    if(id != 0) {
	    text = (char*) malloc (sizeof(char) * filesize);
    }

    MPI::COMM_WORLD.Bcast(text, filesize, MPI::CHAR, 0); // qtd de palavras    

    initDict(); //Todos os nos possuem os mesmo dicionarios
    
    //Passo 1 - Gerar palavras ate 5 letras
    int qtd = shortDict.getQtd();
    int* reduced = (int*) malloc(sizeof(int) * qtd);
    int controle=1;
    int debug = 0;
    tic(); //Conta o tempo para achar as palavras
    for(;;) {
        generateWords(id, chunkSize, debug);

        // Junta as marcacoes das palavras de todos os nos
        //int beforeMarked = shortDict.getQtdMarked();

        MPI::COMM_WORLD.Allreduce(shortDict.getMarked() , reduced, qtd, MPI::INT, MPI::LOR);
  
        shortDict.setMarked(reduced);

        if(shortDict.getQtdMarked() == 8096)
            debug = 1;

        if(id == 0) {
            imprime_prop(controle, shortDict.getQtdMarked(), qtd);
            float prop = shortDict.getQtdMarked() / (float) qtd;
            if(shortDict.getQtdMarked() == 8096) {
                shortDict.print(0);
            }
            
            if(prop*100 >= porcent) {//Calcula somente ateh tal porcentagem
                break;
            }
//          cout << beforeMarked << ":" << shortDict.getQtdMarked() << endl;
        }
    }
    free(reduced);
    //Passo 2 - Concatenar palavras de ate 5 letras para gerar palavras maiores
    /*
    int i,j,k;
    char* combined[2];
                                                                        
    for(i=0;i<total_palavras; i++) {
        if(dict[i].marked == true) {
            for(j=i; j<total_palavras; j++) {
                if(dict[j].marked == true) {
                    combined[0] = mystrcat(dict[i].word, dict[j].word);
                    combined[1] = mystrcat(dict[j].word, dict[i].word);
                                                                        
                    for(k=0; k<2; k++) {
                        if(dict_mark(combined[k]) != -1) {
                            qtd_encontradas++;
                            if(qtd_encontradas == 20000)
                                break;
                            
                                imprime_prop();
                            }
                        }
                        free(combined[k]);
                    }
            }
        }
    }
    */
    MPI::COMM_WORLD.Barrier();
	end = MPI::Wtime();
	MPI::Finalize();

	if(id == 0) {
		cout << "Runtime = " << end - start << endl;
	}

	return 0;
}

//Variaveis globais usadas somente pelas funcoe auxiliares
//char* text; // Armazena todo o texto do arquivo


// Aloca e concatena duas strings
char* mystrcat(const char* str1,const char* str2) {
	char* cat = (char*) malloc(sizeof(char) * (strlen(str1) + strlen(str2) + 1));
	strcpy(cat, str1);
	strcat(cat, str2);
	return cat;
}

//Conta o numero de palavras no arquivo
void count_words() {
    totalPalavras=0;
    int dictShortQtd=0;
    int dictCompoundQtd=0;
    int qtdPrefix=0;

    char *pch = strtok (text, "\n");
    int length;
    
    for(int i=0; i<6; i++)
    	qtd_palavra[i] = 0;
                                                                       
    while(pch != NULL){
        length = strlen(pch);

        if(length <= 5) {
            qtd_palavra[length]++;
            qtd_palavra[0]++;
            dictShortQtd++;
        } else
            dictCompoundQtd++;
        totalPalavras++;
        qtdPrefix+=length-1;

    	pch = strtok (NULL, "\n");                                    
    }

    shortDict.init(dictShortQtd, MAX_SHORT_SIZE);
    compoundDict.init(dictCompoundQtd, MAX_COMPOUND_SIZE);
    cout << qtdPrefix << endl;
}

void readText(const char* filename) {
	FILE *arq_palavras = fopen(filename, "r");

	//Calcula o tamanho do arquivo
	fseek(arq_palavras, 0L, SEEK_END);
	filesize = ftell(arq_palavras);
	fseek(arq_palavras, 0L, SEEK_SET);

	text = (char*) malloc (sizeof(char) * filesize);
	fread(text, 1, filesize, arq_palavras);	

    fclose(arq_palavras);
}

void initDict() {
    char* temp;
    
	temp = (char*) malloc (sizeof(char) * filesize);

    memcpy(temp, text, sizeof(char) * filesize); // Salva o text em temp
    count_words(); //Conta o número de palavras
                                                               
    memcpy(text, temp, sizeof(char) * filesize);                                                           
    parser(); //Parsea o arquivo
                                                               
    memcpy(text, temp, sizeof(char) * filesize); //Restaura o text
    free(temp);
}

// Le todo o arquivo e inicializa os dicionarios
void loadDict(const char* filename) {
    readText(filename);
    
    initDict();
}

// Parsea as palavras lidas
void parser() {
    char *pch = strtok (text, "\n");
    int countShort=0;
    int countCompound=0;
    
    while(pch != NULL){
        if(strlen(pch) <= 5)
            shortDict.insert(countShort++, pch);
        else
            compoundDict.insert(countCompound++, pch);
    	pch = strtok (NULL, "\n");                                    
    }
                                                                         
}

double myrand(struct drand48_data &buffer) {
    double temp;
    drand48_r(&buffer, &temp);
    return temp;
}

// Gera o tamanho de uma palavra de modo uniforme
int gera_tamanho_palavra(struct drand48_data &buffer) {
    int r = myrand(buffer) * qtd_palavra[0];
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
char* gera_palavra(int tamanho, struct drand48_data &buffer)  {
	char* str;

	str = (char*) malloc(sizeof(char) * (tamanho));
	
	for(int i = 0; i < tamanho; i++){
		str[i] = ('a' + myrand(buffer) * 26);
	}
	str[tamanho] = '\0';
	
	return str;
}

//Imprime a proporcao de palavras encontradas
void imprime_prop(int &controle, int qtdMarked, int total) {
    float prop = qtdMarked / (float) total;
	if((prop >= 0.1) && (controle == 1)) {
    	cout << qtdMarked << " - 10% encontrado: " << toc() << endl;
    	controle++;	
    }
    if ((prop >= 0.2) && (controle == 2)) {
    	cout << qtdMarked << " - 20% encontrado: " << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.3) && (controle == 3)) {
    	cout << qtdMarked << " - 30% encontrado: " << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.4) && (controle == 4)) {
    	cout << qtdMarked << " - 40% encontrado: " << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.5) && (controle == 5)) {
    	cout << qtdMarked << " - 50% encontrado: " << toc() << endl;
    	controle++;
    } 
    if ((prop >= 0.6) && (controle == 6)) {
    	cout << qtdMarked << " - 60% encontrado: " << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.7) && (controle == 7)) {
    	cout << qtdMarked << " - 70% encontrado: " << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.8) && (controle == 8)) {
    	cout << qtdMarked << " - 80% encontrado: " << toc() << endl;
    	controle++;
    } 
    if ((prop >= 0.9) && (controle == 9)) {
    	cout << qtdMarked << " - 90% encontrado: " << toc() << endl;
    	controle++;
    } 
    if (prop == 1.0) {
    	cout << qtdMarked << " - 100% encontrado: " << toc() << endl;
	}
}

void seed_rand(int thread_n, struct drand48_data *buffer)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec * thread_n + tv.tv_usec, buffer);
}

char *mysubstr(char *str, int begin, int end) {
    int length = end - begin + 1;
    char* sub = (char*) malloc(sizeof(char) * (length + 1));
    strncpy(sub, str+begin, length);
    sub[length] = '\0';
    return sub;
}

//Gera palavras com ate 5 letras de forma aleatoria
void generateWords(int id, int qtd, int debug) {
    int qtd_encontradas=0;
    #pragma omp parallel default(none) shared(id, qtd_encontradas, qtd, shortDict, qtd_palavra, debug, cout) 
    {
        int thread_id = omp_get_thread_num();
        struct drand48_data drand_buffer;
    	seed_rand(id*17 + thread_id*13, &drand_buffer);
        int tamanho;
        char new_word[6];

        int gerar_tamanho[6]; //Indica se ainda precisamos gerar tal tamanho
        for(int i=1; i<6; i++) {
            gerar_tamanho[i] = !(shortDict.getQtdMarkedLength(i) == qtd_palavra[i]);
            if(id == 0) 
                if(debug == 1)
                    if(gerar_tamanho[i] == 1)
                        cout << "tamanhos restantes: " << i << endl;
            //cout << "gerar tamanho? " << i << ":" << gerar_tamanho[i] << endl;
        }

        #pragma omp for reduction(+:qtd_encontradas) private(tamanho, new_word)
        for(int i=0; i<qtd; i++) {
            // So gera tamanho de palavras ainda nao encontradas
            
            do {
                tamanho = gera_tamanho_palavra(drand_buffer);// myrand(drand_buffer) * 5 + 1;
                //cout << tamanho << " " << gerar_tamanho[tamanho] << endl;
            } while(!gerar_tamanho[tamanho]);
            
            
            //tamanho = gera_tamanho_palavra(drand_buffer);
            for(int j=0; j<tamanho; j++)
		        new_word[j] = ('a' + myrand(drand_buffer) * 26);
            new_word[tamanho] = '\0';

//            char* new_word = gera_palavra(tamanho, drand_buffer);
//            cout << new_word << endl;
           
            if(shortDict.markWord(new_word) != NOT_FOUND) {
    			qtd_encontradas++;
            }
                                                                            
 //           free(new_word);
        }
    }
}
