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
#include "mpi.h"

#include "utils.h"
#include "dict.hpp"

#define TAG_SHORT 7
#define GENERATE 1000

using namespace std;

Dict shortDict;
Dict compoundDict;

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int totalPalavras; //Total de palavras a serem encontradas
int total_prop;
//int qtd_encontradas=0; //Total de palavras encontradas

int controle=1;
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
void readText(const char* filename);


int main(int argc, char* argv[]) {
    MPI::Status mpi_status;
	double start, end;

    MPI::Init(argc, argv);

    int id = MPI::COMM_WORLD.Get_rank();
    int numprocs = MPI::COMM_WORLD.Get_size();

	MPI::COMM_WORLD.Barrier();
	start = MPI::Wtime();

	/*
	
	Carrega o dicionario em todos os nos
	
	*/

    char **shortWords;
    char** recvBuffer;
    int qtd;
    int maxlenth;
	readText(argv[1]);
    if(id == 0) {

    	end = MPI::Wtime();

		cout << "load dict time: " << end - start << endl;
/*
        shortWords = shortDict.getWords();
        for(int i=0; i<5; i++)
            cout << shortWords[i] << endl;
*/
        for(int i=0; i<10; i++)
            cout << text[i];
        cout << endl;
/*
        //master();
        //shortWords = shortDict.getWords();
        qtd = shortDict.getQtd() / numprocs;
        maxlenth = shortDict.getMaxWordLength();
        */
    }

    //Envia o tamanho do buffer para todos os processadores
   // MPI::COMM_WORLD.Bcast(&filesize, 1, MPI::INT, 0); // qtd de palavras

    // Aloca o text para os outros processos
    //if(id != 0) {
	//    text = (char*) malloc (sizeof(char) * filesize);
    //}

    //MPI::COMM_WORLD.Bcast(text, filesize, MPI::CHAR, 0); // qtd de palavras

    cout << id << endl;
    for(int i=0; i<10; i++)
        cout << text[i];
    cout << endl;


    //MPI::COMM_WORLD.Bcast(&qtd, 1, MPI::INT, 0); // qtd de palavras
    //MPI::COMM_WORLD.Bcast(&maxlenth, 1, MPI::INT, 0); //tamanho maximo delas
/*
    qtd = 2;
    maxlenth = 10;

    //Aloca o recvBuffer
    recvBuffer = (char**) malloc(sizeof(char*) * qtd);
    for(int i=0; i<qtd; i++)
        recvBuffer[i] = (char*) malloc(sizeof(char) * maxlenth);

    char **r;
    r = (char**) malloc(sizeof(char*) * qtd);
    for(int i=0; i<2; i++)
        r[i] = (char*) malloc(sizeof(char) * maxlenth);

//    MPI::COMM_WORLD.Scatter(shortWords, qtd*maxlenth, MPI::CHAR, recvBuffer, qtd*maxlenth, MPI::CHAR, 0);
    MPI::COMM_WORLD.Scatter(&(teste[0][0]), qtd*maxlenth, MPI::CHAR, &(r[0][0]), qtd*maxlenth, MPI::CHAR, 0);


    for(int i=0; i<2; i++) {
        r[i][maxlenth-1] = '\0';
        cout << id << " - " << r[i] << " " << strlen(r[i]) <<  endl;
    }
*/
/*
    for(int i=0; i<5; i++)
        cout << id << ":" << recvBuffer[i] << endl;
*

    for(int i=0; i<=qtd; i++)
        free(recvBuffer[i]);
    free(recvBuffer);
*/
//        MPI::COMM_WORLD.Scatter(shortWords, sendcount, MPI::CHAR, recvBuffer, recvcount, MPI::CHAR, 0); 

//        MPI::COMM_WORLD.Bcast(shortWords, shortDict.getQtd() * shortDict.getMaxWordLength(), MPI::CHAR, 0, TAG_SHORT); 

		// Recebe as palavras encontradas pelos nos 1,2 e 3
        /*
        for(int i=1; i<=3; i++) {
            MPI::COMM_WORLD.Recv(idx_encontrados[i-1], GENERATE, MPI::INT, i, TAG_SHORT);
			
            cout << "recv " << i << endl;
            for(int j=0; j<GENERATE; j++) {
                qtd_encontradas += shortDict.markPos(idx_encontrados[i-1][j]);
			}
        }
		cout << "Total marked: " << qtd_encontradas << endl;
        */
        /* 
        
        Thread 0 se comunica com os outros nos
		Thread 1 gera palavras de 1 a 3 letras
		Thread 2 gera palavras de 4 letras
		Thread 3 gera palavras de 5 letras
        
        
        */
		/*
        #pragma omp parallel default(none)
		{
			int thread_id = omp_get_thread_num();
			struct drand48_data drand_buffer;
			seed_rand(thread_id, &drand_buffer);

	//		while(true) {
				
				
				Thread 0 se comunica com os outros nos 
				
				
				
			}
		}
		//cout << id << ":" << shortDict.getQtd() << "-" << compoundDict.getQtd() << endl;
        */
//    } else {
        /*
        if(id == 1) {
            char t[3][20];
            MPI::COMM_WORLD.Recv(t, 3*20, MPI::CHAR, 0, TAG_SHORT);
            for(int i=0; i<3; i++)
                cout << t[i] << endl;
        }*/
        //worker();
		
        //Node 1 gera palavras de tamanho 1 a 3
        //Node 2 gera palavras de tamanho 4
        //Node 3 gera palavras de tamanho 5
/*
        int qtd_encontradas=0;
        int tamanho;
		int idx[GENERATE];

        if(id <= 3) {

            int idx[GENERATE];
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                struct drand48_data drand_buffer;
				seed_rand(id*17 + thread_id*13, &drand_buffer);
                #pragma omp for reduction(+:qtd_encontradas) private(tamanho)
                for(int i=0; i<GENERATE; i++) {
                    if(id == 1) {
                        tamanho = myrand(drand_buffer) * 3 + 1; //Tamanhos de 1 a 3
                    } else {
                        tamanho = id + 2; // no 2 - tamanho 4, no 3 - tamanho 5
                    }

                    char* new_word = gera_palavra(tamanho, drand_buffer);
                   
                    idx[i] = shortDict.markWord(new_word);
					if(idx[i] != NOT_FOUND)
						qtd_encontradas++;

                    free(new_word);
                }
            }
			cout << id << " marked: " << qtd_encontradas << endl;
			
			MPI::COMM_WORLD.Send(idx, GENERATE, MPI::INT, 0, TAG_SHORT);
        }*/
//    }
    

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

    char *pch = strtok (text, "\n");
    int length;
    
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
    count_words(); //Conta o número de palavras
                                                               
    memcpy(text, temp, sizeof(char) * filesize);
                                                               
    parser(); //Parsea o arquivo
                                                               
    memcpy(text, temp, sizeof(char) * filesize); //Restaura o text
    free(temp);
}

// Le todo o arquivo e inicializa os dicionarios
void loadDict(const char* filename) {
    char* temp;
	
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

// Gera o tamanho de uma palavra de modo uniforme
int gera_tamanho_palavra(struct drand48_data &buffer) {
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

double myrand(struct drand48_data &buffer) {
    double temp;
    drand48_r(&buffer, &temp);
    return temp;
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
/*
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
*/
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
