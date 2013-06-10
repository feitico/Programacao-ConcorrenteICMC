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

#define CHUNK_SIZE_DEFAULT 100000

using namespace std;

Dict shortDict;
Dict compoundDict;

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int totalPalavras; //Total de palavras a serem encontradas
//int qtd_encontradas=0; //Total de palavras encontradas

char* text;
int filesize; // tamanho do arquivo

/* Funcoes auxiliares */
int gera_tamanho_palavra(unsigned short xsubi[3]);// Gera o tamanho de uma palavra de forma uniforme
void parser();// Parsea as palavras lidas

void readText(const char* filename); //Le o arquivo inteiro
void initDict(); // Inicializa o dicionario

void generateWords(int id, int qtd); //Gera palavras com ate 5 letras de forma aleatoria
void imprime_prop(int &controle, int qtdMarked, int total);

int main(int argc, char* argv[]) {
    int porcent; //Porcentagem a ser calculada
    int chunkSize; //Gera tantas palavras aleatorias
    int id; // Id do no
    int numprocs; // Numero de nos
    int qtd; // Qtd de palavras nos dicionarios
    int* reduced; // Utilizado para o MPI_Allreduce
    int controle; // controla a impressao das porcentagem concluidas
    char** compoundWords; //Armazena as palavras compostas
    char* currentWord; //Palavra atual a ser composta
    int length; //Tamanho da palavra atual a ser composta
    int begin; //Posicao inicial para a composicao da palavra
    int substrLen; //Tamanho da substr a compor uma palavra maior
    char substr[6]; //Armazena a substring da palavra a ser composta
    int tamanho_bloco; //Tamanho do bloco de palavras que seram compostas por um no
    int inicio_bloco; //Inicio do bloco para um determinado no
    int fim_bloco; //Fim do bloco para um determinado no

    if(argc != 3) {
        cout << "Uso: " << argv[0] << " palavras.txt porcentagem" << endl;
        exit(-1);
    }
    porcent = atoi(argv[2]);
    chunkSize = CHUNK_SIZE_DEFAULT;

    MPI::Init(argc, argv);
    srand(time(NULL));

    id = MPI::COMM_WORLD.Get_rank();
    numprocs = MPI::COMM_WORLD.Get_size();

	MPI::COMM_WORLD.Barrier();
	tic(); // Marca o tempo inicial
    
    if(id == 0) {
	    readText(argv[1]); //Somente o master le o arquivo inteiro

//		cout << "read file time: " << toc() << endl;
    }

    //Envia o tamanho do buffer para todos os processadores
    MPI::COMM_WORLD.Bcast(&filesize, 1, MPI::INT, 0); // qtd de palavras

    // Aloca o text para os outros processos
    if(id != 0) {
	    text = (char*) malloc (sizeof(char) * filesize);
    }

    //Envia o conteudo do texto para todos os nos
    MPI::COMM_WORLD.Bcast(text, filesize, MPI::CHAR, 0); // qtd de palavras    
    //Inicializa o dicionario
    initDict(); //Todos os nos possuem os mesmo dicionarios

    //Passo 1 - Gerar palavras de ate 5 letras
    qtd = shortDict.getQtd();
    reduced = (int*) malloc(sizeof(int) * qtd);
    controle=1;
	/*********************************************************************
	Passo 1 - Gerar palavras com ate 5 letras
	Todos os nos geram palavras aleatorias
	É utilizado a operacao ALLREDUCE com a operação OR,
	desse modo, apos chunkSize de palavras geradas todos os nos sabem
	quais palavras ja foram geradas
	**********************************************************************/
    for(;;) {
        generateWords(id, chunkSize); // Gera n palavras
        // Realiza um OR com todos os blocos marcados
        MPI::COMM_WORLD.Allreduce(shortDict.getMarked(), reduced, qtd, MPI::INT, MPI::LOR);
 
        // Realiza a marcacao em todos os dicionarios
        shortDict.setMarked(reduced);

        // Master - Imprime a porcentagem ja alcancada de palavras com ate 5 letras
        if(id == 0) {
            imprime_prop(controle, shortDict.getQtdMarked(), qtd);
		}
		float prop = shortDict.getQtdMarked() / (float) qtd;
           
        if(prop*100 >= porcent) {//Calcula somente ateh tal porcentagem
        	break;
        }
    }
    free(reduced);

	/*********************************************************************
	Passo 2 - Compor palavras de mais de 5 letras

    Cada no ira trabalhar em um bloco de palavras especifico, no final
    utiliza-se o AllREDUCE com a operacao OR para que todos os nos saibam
    quais palavras ja foram marcadas
	**********************************************************************/
    controle=1;
    compoundWords = compoundDict.getWords();
    begin=0;
    substrLen=5;
    qtd=compoundDict.getQtd();
    tamanho_bloco = qtd / numprocs + 1;
    inicio_bloco = id * tamanho_bloco;
    fim_bloco = inicio_bloco + tamanho_bloco;
    /********************************************************************
    Algoritmo de Composicao de palavras
    Comeca no comeca da palavra a ser composta
    Procura sempre casar com a substring mais longa
    *********************************************************************/
    for(int i=inicio_bloco;i<fim_bloco; i++) {
        if(i >= qtd)
            break;
        currentWord = compoundWords[i];
        length = strlen(currentWord);
        begin=0;
        substrLen=5;
        for(;;) {
            memcpy(substr, currentWord + begin, substrLen);
            substr[substrLen] = '\0';
          
            if(shortDict.search(substr) != NOT_FOUND) {
                begin = begin + substrLen;
                if(begin+5 >= length)
                    substrLen = length - begin;
                else
                    substrLen = 5;
            } else {
                substrLen--;
                if(substrLen == 0) {
                    begin++;
                    if(begin+5 >= length)
                        substrLen = length - begin;
                    else
                        substrLen = 5;
                }
            }
            if(begin >= length) {
                compoundDict.markPos(i);
                break;
            }
        }

        if(id == 0)
            imprime_prop(controle, compoundDict.getQtdMarked() * numprocs, compoundDict.getQtd());

        float prop = compoundDict.getQtdMarked()*numprocs / (float) compoundDict.getQtd();
        
        if(prop*100 >= porcent) {//Calcula somente ateh tal porcentagem
            break;
        }
    }

    /*
     * Informa todos os nos quais palavras ja foram encontradas
     *
     * **/
    reduced = (int*) malloc(sizeof(int) * qtd);
    // Realiza um OR com todos os blocos marcados
    MPI::COMM_WORLD.Allreduce(compoundDict.getMarked(), reduced, qtd, MPI::INT, MPI::LOR);                                                                                
    // Realiza a marcacao em todos os dicionarios
    compoundDict.setMarked(reduced);

    free(reduced);
    //}

	MPI::COMM_WORLD.Barrier();	
	MPI::Finalize();
	// Master - Imprime o tempo total de execucao
	if(id == 0) {
//		cout << "Runtime = " << toc() << endl;

        cout << toc() << endl;
	}	

	return 0;
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
    //cout << qtdPrefix << endl;
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
int gera_tamanho_palavra(unsigned short xsubi[3]) {
    int r = nrand48(&xsubi[0]) % qtd_palavra[0];
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
char* gera_palavra(int tamanho, unsigned short xsubi[3])  {
	char* str;

	str = (char*) malloc(sizeof(char) * (tamanho));
	
	for(int i = 0; i < tamanho; i++){
		str[i] = ('a' + nrand48(&xsubi[0]) % 26);
	}
	str[tamanho] = '\0';
	
	return str;
}
/*
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
*/
void imprime_prop(int &controle, int qtdMarked, int total) {
    float prop = qtdMarked / (float) total;
	if((prop >= 0.1) && (controle == 1)) {
    	cout << toc() << endl;
    	controle++;	
    }
    if ((prop >= 0.2) && (controle == 2)) {
    	cout << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.3) && (controle == 3)) {
    	cout << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.4) && (controle == 4)) {
    	cout << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.5) && (controle == 5)) {
    	cout << toc() << endl;
    	controle++;
    } 
    if ((prop >= 0.6) && (controle == 6)) {
    	cout << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.7) && (controle == 7)) {
    	cout << toc() << endl;
    	controle++;	
    } 
    if ((prop >= 0.8) && (controle == 8)) {
    	cout << toc() << endl;
    	controle++;
    } 
    if ((prop >= 0.9) && (controle == 9)) {
    	cout << toc() << endl;
    	controle++;
    } 
    if (prop == 1.0) {
    	cout << toc() << endl << endl;
	}
}















//Gera palavras com ate 5 letras de forma aleatoria
void generateWords(int id, int qtd) {
    int qtd_encontradas=0;
    #pragma omp parallel default(none) shared(id, qtd_encontradas, qtd, shortDict, qtd_palavra, cout) 
    {
        int tamanho;
        char new_word[6];

        int gerar_tamanho[6]; //Indica se ainda precisamos gerar tal tamanho
        for(int i=1; i<6; i++) {
            gerar_tamanho[i] = !(shortDict.getQtdMarkedLength(i) == qtd_palavra[i]);
        }
        unsigned short seed[3] = {rand() % 1024, rand() % 1024, rand() % 1024};
        #pragma omp parallel for reduction(+:qtd_encontradas) private(tamanho, new_word)
        for(int i=0; i<qtd; i++) {
            // So gera tamanho de palavras ainda nao encontradas   
            do {
                tamanho = gera_tamanho_palavra(seed);// myrand(drand_buffer) * 5 + 1;
            } while(!gerar_tamanho[tamanho]);
            
            // Gera uma palavra de tamanho n aleatoriamente
            for(int j=0; j<tamanho; j++)
		        new_word[j] = ('a' + nrand48(&seed[0]) % 26);
            new_word[tamanho] = '\0';

            if(shortDict.markWord(new_word) != NOT_FOUND) {
    			qtd_encontradas++;
            }
                                                                            
        }
    }
}

