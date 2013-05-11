#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include "omp.h"

#define MAX_NUM 8000
#define RAIZ_MAX_NUM 90

#define SMALL 0
#define LARGE 1

using namespace std;

int primo[MAX_NUM]; //Vetor que indica se um numero eh primo ou nao

// Funcoes auxiliares
void init_crivo(int n); // Inicializa o crivo de erastotenes 
int isPrimo(int num); // Retorna 1 se for primo
int palindromo(string str); // Retorna 1 se for palindromo 
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string

int main(int argc, char* argv[]) {
	if(argc != 4) {
		printf("Usage: ./unified type nthreads entrada.txt\n");
		exit(-1);
	}
	
    int n_threads = atoi(argv[2]);
    string str;
    int flag;

    omp_set_num_threads(n_threads);

    #pragma omp master
    {
        ifstream in(argv[3]);
        int thread=0;
 
        //Le uma string e habilita a thread i a processa-la
        while(!in.eof()) {
            in >> str;
            #pragma omp flush   
            flag = 1;
            #pragma omp flush (flag)
            cout << "master: " << str << endl;
            thread = (thread+1) % n_threads;
        }

        in.close();
    } 

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        string mystr;
        #pragma omp flush (flag)
        while(flag != 1) {
            #pragma omp flush (flag)    
        }
        #pragma omp flush
        mystr = str;
        cout << id  << " " << mystr << endl;
    }

	return 0;
}

// Inicializa o crivo de erastotenes 
void init_crivo(int n) {
	int i, j;
	for(i=2; i<n; i++)
		primo[i] = i;
	
	int raiz = sqrt(n)+1;
	for(i=2; i<=raiz; i++) {
		if(primo[i] == i) {
			for(j=i+i; j<MAX_NUM; j+=i)
				primo[j] = 0;
		}
	}
}

// Retorna 1 se for primo
int isPrimo(int num) {
	return primo[num];
}

// Retorna 1 se for palindromo 
int palindromo(string str) {
	int i;
	int half;
	int length = str.size();

	if(length == 1)
		return 0;

	if(length % 2 == 0)
		half = length / 2;
	else
		half = (length / 2) + 1;


	for(i=0; i<=half; i++) {
		if(tolower(str[i]) != tolower(str[length-i-1])) {
			return 0;
		}
	}

	return 1;
}

// Remove os espacos em branco a direita e a esquerda da string
string trim(const string &str)
{
	size_t s = str.find_first_not_of(" \n\r\t");
	size_t e = str.find_last_not_of (" \n\r\t");

	if(( string::npos == s) || ( string::npos == e))
		return "";
	else
		return str.substr(s, e-s+1);
}
