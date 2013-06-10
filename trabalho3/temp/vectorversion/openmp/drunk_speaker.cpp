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

using namespace std;

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


// Estrutura que representa se uma palavra foi encontrada ou nao
struct entry {
    char* word;
    bool marked;
};


/* Variaveis globais */
entry* dict; // Representa as nossas palavras
FILE *arq_palavras; //Ponteiro para o arquivo
char* text; // Armazena todo o texto do arquivo

int qtd_palavra[6]; //Qtd de palavras de tamanho 1,2,3,4,5 ( usado para gerar uniformimente essas palavras)
int total_palavras; //Total de palavras a serem encontradas
int total_prop;
int qtd_encontradas=0; //Total de palavras encontradas

clock_t begin; //Utilizado pelas funcoes tic() e toc() para calcular marcar o tempo inicial
int controle=1;

/* Funcoes auxiliares */
void tic(); //Marca o tempo de inicio
double toc();//Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
void ler_arquivo();// Le todo o arquivo
void calc_proporcao();// Calcula a proporcao de palavras com ateh 5 letras
void calc_total_palavras();//Calcula total de palavras
void init_dict(const char* filename);// Inicializa o dicionario
int gera_tamanho_palavra(unsigned int &myseed, struct drand48_data &buffer);// Gera o tamanho de uma palavra de forma uniforme
char* gera_palavra(int tamanho, unsigned int &myseed, struct drand48_data &buffer);// Gera uma palavra com um certo tamanho
void imprime_prop(); //Imprime a proporcao de palavras encontradas
void parser();// Parsea as palavras lidas
char* mystrcat(const char* str1,const char* str2);
int entryCompare( const void * a, const void * b); //Compara duas entrys
int dict_mark(const char* word); // Marca uma palavra no dicionario
void seed_rand(int thread_n, struct drand48_data *buffer);


int main(int argc, char* argv[]) {
	//MPI::Status mpi_status; // status do MPI_Recv

	if(argc != 4) {
		cout << "Uso: " << argv[0] << " palavras.txt n thread" << endl;
		exit(-1);
	}
	int num = atoi(argv[2]);
	int threads = atoi(argv[3]);

/*
	MPI::Init(argc, argv); //Inicializa o MPI
	int host = MPI::COMM_WORLD.Get_rank(); // Identifica o host
	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		cout << "Host: " << host << " - Thread: " << thread_id << ": Hello World" << endl;
	}
	MPI::Finalize();
*/


	omp_set_num_threads(threads);

	//Inicializa o dicionario
	init_dict(argv[1]);	
	//Gera as palavras com até cinco letras
    total_prop = qtd_palavra[0];
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
//						cout <<  thread_id << ":" << new_word << endl;
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
	return -1;
    controle =1;
    total_prop = total_palavras;

    int i,j,k;
    char* combined[2];
	int chunk_init = 0;
	int chunk_size = total_palavras / 1000;
	int chunk_end = chunk_init + chunk_size;

	while(true) {

        #pragma omp parallel for private(i,j,k, combined) shared(dict) reduction(+ : qtd_encontradas, generate)
        for(i=chunk_init; i<chunk_end; i++) {
            if(dict[i].marked == true) {
                for(j=i; j<total_palavras; j++) {
                    if(dict[j].marked == true) {
                        combined[0] = mystrcat(dict[i].word, dict[j].word);
                        combined[1] = mystrcat(dict[j].word, dict[i].word);

						generate++;
						cout << omp_get_thread_num() << "-" << i << ":" << j << endl; 
                        for(k=0; k<2; k++) {
                            if(dict_mark(combined[k]) != -1)
                                qtd_encontradas++;
                            free(combined[k]);
                        }
                    }
                }
            }
        }
		chunk_init = chunk_end+1;
		chunk_end = chunk_init + chunk_size;
		
		cout << generate << ":" << qtd_encontradas << endl;
		imprime_prop();
        float prop = qtd_encontradas / (float) total_palavras;
        if(prop >= 0.6)
            break;
    } 

    cout << "Total: " << qtd_encontradas << endl;

    int mark=0;
    int notmark=0;
#pragma omp parallel for reduction(+:mark, notmark)
    for(int i=0; i<total_palavras; i++)
        if(dict[i].marked == true)
            mark++;
        else
            notmark++;

    cout << mark << endl;
    cout << notmark << endl;

    
    fclose(arq_palavras);
    free(dict);
	cout << "elapsed time: " << toc() << endl;

	return 0;
}

// Aloca e concatena duas strings
char* mystrcat(const char* str1,const char* str2) {
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

//Conta o numero de palavras no arquivo
void count_words() {
    total_palavras=0;
    while (EOF != (fscanf(arq_palavras, "%*[^\n]"), fscanf(arq_palavras, "%*c")))
            ++total_palavras;
}

// Le todo o arquivo
void ler_arquivo(const char* filename) {
	arq_palavras = fopen(filename, "r");

	//Calcula o tamanho do arquivo
	fseek(arq_palavras, 0L, SEEK_END);
	int size = ftell(arq_palavras);
	fseek(arq_palavras, 0L, SEEK_SET);

    count_words(); //Conta o número de palavras

	fseek(arq_palavras, 0L, SEEK_SET); //Volta para o comeco do arquivo

	text = (char*) malloc (sizeof(char) * size);
	fread(text, 1, size, arq_palavras);	
}

// Inicializa o dicionario
void init_dict(const char* filename) {
	tic();
	ler_arquivo(filename);
	cout << "leitura: " << toc() << endl;

    //Aloca o dicionario
    dict = (entry*) malloc(sizeof(struct entry) * (total_palavras+1));

	tic();
	parser();
	cout << "parser: " << toc() << endl;
}

// Parsea as palavras lidas
void parser() {
    char *pch = strtok (text, "\n");
    int length;
    int count=0;
    
    for(int i=0; i<6; i++)
    	qtd_palavra[i] = 0;

    while(pch != NULL){
	    dict[count].word = strdup(pch);
        dict[count++].marked = false;
        
        length = strlen(pch);

        if(length <= 5) {
            qtd_palavra[length]++;
            qtd_palavra[0]++;
        }
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

int binary_search(const char* word) {
    int imin = 0;
    int imax = total_palavras-1;

    // Continua buscando enquanto [imin, imax] nao eh vazio
    while(imax >= imin) {
        int imid = (imin + imax) / 2;

        // Decide em qual sub array para procurar
        if(strcmp(dict[imid].word, word) < 0)
            imin = imid + 1;
        else if (strcmp(dict[imid].word, word) > 0)
            imax = imid - 1;
        else
            return imid;
    }
    return -1;
}

int dict_mark(const char* word) {
    int indice = binary_search(word);
    if(indice != -1) {
        if(dict[indice].marked == true)
            return -1;
        else {
            dict[indice].marked = true;
            return indice;
        }
    } else
        return -1;
}

int entryCompare( const void * a, const void * b){
    entry* e1 = (entry*) a;
    entry* e2 = (entry*) b;

    return strcmp(e2->word, e1->word) < 0;
}


void seed_rand(int thread_n, struct drand48_data *buffer)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec * thread_n + tv.tv_usec, buffer);
}
