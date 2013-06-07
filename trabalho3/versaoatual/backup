#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <map>
#include <iostream>
#include <set>
#include "omp.h"
#include <sys/time.h>
//#include "mpi.h"

#include "utils.h"
#include "dict.hpp"

using namespace std;

Dict shortDict;
Dict compoundDict;

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int totalPalavras; //Total de palavras a serem encontradas
int total_prop;
int qtd_encontradas=0; //Total de palavras encontradas

int controle=1;

/* Funcoes auxiliares */
void readDictFile();// Le todo o arquivo
void calc_proporcao();// Calcula a proporcao de palavras com ateh 5 letras
void calc_totalPalavras();//Calcula total de palavras

void loadDict(const char* filename);// Inicializa o dicionario


int gera_tamanho_palavra(unsigned int &myseed, struct drand48_data &buffer);// Gera o tamanho de uma palavra de forma uniforme
char* gera_palavra(int tamanho, unsigned int &myseed, struct drand48_data &buffer);// Gera uma palavra com um certo tamanho
void imprime_prop(); //Imprime a proporcao de palavras encontradas
void parser();// Parsea as palavras lidas
char* mystrcat(const char* str1,const char* str2);
int entryCompare( const void * a, const void * b); //Compara duas entrys
int dict_mark(const char* word); // Marca uma palavra no dicionario
void seed_rand(int thread_n, struct drand48_data *buffer);
char *mysubstr(char *str, int begin, int end);
int binary_search(char** dict, const char* word);


int main(int argc, char* argv[]) {
	/*
    if(argc != 4) {
		cout << "Uso: " << argv[0] << " palavras.txt n thread" << endl;
		exit(-1);
	}
	int num = atoi(argv[2]);
	int threads = atoi(argv[3]);
	//MPI::Status mpi_status; // status do MPI_Recv
    int id;
    int numprocs;
    */
    MPI::Status mpi_status;

    loadDict(argv[1]);

/*
    char str[] = "festabreganocaaso";
    int length = strlen(str);
    char* sub[3];

    for(int i=0; i<length-2; i++) {
        for(int j=i+1; j<length-1; j++) {
            sub[0] = mysubstr(str, 0, i);
            sub[1] = mysubstr(str, i+1, j);
            sub[2] = mysubstr(str, j+1, length);

            cout << sub[0] << " - " << sub[1] << " - " << sub[2] << endl;

            free(sub[0]);
            free(sub[1]);
            free(sub[2]);
        }
    }
*/

/*
	MPI::Init(argc, argv); //Inicializa o MPI
	id = MPI::COMM_WORLD.Get_rank(); // Identifica o host
	numprocs = MPI::COMM_WORLD.Get_size();

    //Master Processor
    if(id == 0) {
        cout << id << " Lets work " << omp_get_thread_num() << endl;
        //Inicializa o dicionario
        loadDict(argv[1]);

        //Gera as palavras com até cinco letras
        total_prop = dictShortQtd;
        tic();
        int generate=0;
            
            while(true) {
                #pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();
                    unsigned int myseed = thread_id * time(NULL);
                    struct drand48_data drand_buffer;
                    seed_rand(thread_id, &drand_buffer);
                    #pragma omp parallel for reduction(+:qtd_encontradas, generate)
                    for(int i=0; i<num; i++) {
                        int tamanho = gera_tamanho_palavra(myseed, drand_buffer);
                        char *new_word = gera_palavra(tamanho, myseed, drand_buffer);
                        generate++;
                        //cout << thread_id << ":" << myseed << endl;
                        //cout <<  thread_id << ":" << new_word << endl;

                        if(dict_mark(new_word) != -1) {
                            qtd_encontradas++;
//                          cout <<  thread_id << ":" << new_word << endl;
                        }

                        free(new_word);
                    }

                }
                //cout << qtd_encontradas << endl;
                imprime_prop();
                float prop = qtd_encontradas / (float) qtd_palavra[0];
                if(prop >= 0.6)
                    break;
            }
        
        cout << "Total: " << qtd_encontradas << endl;
  /*      
        fclose(arq_palavras);
        free(dict);

    } else { // Worker Processors
        cout << id << " NOWORKER" << endl;	
        //return -1;
        controle =1;
        total_prop = totalPalavras;

        int i,j,k;
        char* combined[2];


        while(true) {
            #pragma omp parallel for private(i,j,k, combined) shared(dict) reduction(+ : qtd_encontradas, generate)
            for(i=0; i<totalPalavras; i++) {
                if(dict[i].marked == true) {
                    for(j=i; j<totalPalavras; j++) {
                        if(dict[j].marked == true) {
                            combined[0] = mystrcat(dict[i].word, dict[j].word);
                            combined[1] = mystrcat(dict[j].word, dict[i].word);

                            generate++;
                            
                            for(k=0; k<2; k++) {
                                if(dict_mark(combined[k]) != -1) {
                                    qtd_encontradas++;
                                    cout << omp_get_thread_num() << "-" <<  qtd_encontradas << endl;
                                }
                                free(combined[k]);
                            }
                        }
                    }
                }
            }
        } 

        cout << "Total: " << qtd_encontradas << endl;

        int mark=0;
        int notmark=0;
#pra    gma omp parallel for reduction(+:mark, notmark)
        for(int i=0; i<totalPalavras; i++)
            if(dict[i].marked == true)
                mark++;
            else
                notmark++;

        cout << mark << endl;
        cout << notmark << endl;

    }

	MPI::Finalize();
*/
	cout << "elapsed time: " << toc() << endl;
	return 0;
}

//Variaveis globais usadas somente pelas funcoe auxiliares
char* text; // Armazena todo o texto do arquivo


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

    char *temp = text;
    char *pch = strtok (text, "\n");
    int length;
    int count=0;
    
    for(int i=0; i<6; i++)
    	qtd_palavra[i] = 0;
                                                                       
    while(pch != NULL){
        length = strlen(pch);

        if(length <= 5) {
            qtd_palavra[length]++;
            dictShortQtd++;
        } else
            dictCompoundQtd++;
        totalPalavras++;

    	pch = strtok (NULL, "\n");                                    
    }

    shortDict.init(dictShortQtd, MAX_SHORT_SIZE);
    compoundDict.init(dictCompoundQtd, MAX_COMPOUND_SIZE);

    cout << dictShortQtd << endl << dictCompoundQtd << endl << totalPalavras << endl;
    text = temp;
}

// Le todo o arquivo
void readDictFile(const char* filename) {
	FILE *arq_palavras = fopen(filename, "r");

	//Calcula o tamanho do arquivo
	fseek(arq_palavras, 0L, SEEK_END);
	int size = ftell(arq_palavras);
	fseek(arq_palavras, 0L, SEEK_SET);

	text = (char*) malloc (sizeof(char) * size);
	fread(text, 1, size, arq_palavras);	

	fseek(arq_palavras, 0L, SEEK_SET); //Volta para o comeco do arquivo 
    count_words(); //Conta o número de palavras

	fseek(arq_palavras, 0L, SEEK_SET);

	fread(text, 1, size, arq_palavras);	

    fclose(arq_palavras);
}

// Inicializa o dicionario
void loadDict(const char* filename) {
	tic();
	readDictFile(filename);
	cout << "leitura: " << toc() << endl;

	tic();
	parser();
	cout << "parser: " << toc() << endl;
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
                                                                         
    free(text);
}

// Gera o tamanho de uma palavra de modo uniforme
int gera_tamanho_palavra(unsigned int &myseed, struct drand48_data &buffer) {
	//int r = rand_r(&myseed) % qtd_palavra[0];
	//int r = rand() % qtd_palavra[0];
	double temp;
	drand48_r(&buffer, &temp);
	int r = temp * qtd_palavra[0];
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
char* gera_palavra(int tamanho, unsigned int &myseed, struct drand48_data &buffer)  {
	char* str;
	double temp;

	str = (char*) malloc(sizeof(char) * (tamanho));
	
	for(int i = 0; i < tamanho; i++){
		//str[i] = ('a' + rand_r(&myseed) % 26);
		drand48_r(&buffer, &temp);
		//str[i] = ('a' + rand() % 26);
		str[i] = ('a' + temp * 26);
	}
	str[tamanho] = '\0';
	
	return str;
}

//Imprime a proporcao de palavras encontradas
void imprime_prop() {

    float prop = qtd_encontradas / (float) total_prop;
	if((prop >= 0.1) && (controle == 1)) {
    	cout << qtd_encontradas << " - 10% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.2) && (controle == 2)) {
    	cout << qtd_encontradas << " - 20% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.3) && (controle == 3)) {
    	cout << qtd_encontradas << " - 30% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.4) && (controle == 4)) {
    	cout << qtd_encontradas << " - 40% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.5) && (controle == 5)) {
    	cout << qtd_encontradas << " - 50% encontrado: " << toc() << endl;
    	controle++;
    } else if ((prop >= 0.6) && (controle == 6)) {
    	cout << qtd_encontradas << " - 60% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.7) && (controle == 7)) {
    	cout << qtd_encontradas << " - 70% encontrado: " << toc() << endl;
    	controle++;	
    } else if ((prop >= 0.8) && (controle == 8)) {
    	cout << qtd_encontradas << " - 80% encontrado: " << toc() << endl;
    	controle++;
    } else if ((prop >= 0.9) && (controle == 9)) {
    	cout << qtd_encontradas << " - 90% encontrado: " << toc() << endl;
    	controle++;
    } else if (prop == 1.0) {
    	cout << qtd_encontradas << " - 100% encontrado: " << toc() << endl;
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
