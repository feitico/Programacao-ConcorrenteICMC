#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>

#include "mpi.h"

#define MAX_NUM 8000
#define RAIZ_MAX_NUM 90
#define SMALL 0
#define LARGE 1

using namespace std;

int primo[MAX_NUM];
int atual_proc=0; //id do processador atual
int numprocs; // Quantidade de processadores

// Funcoes auxiliares
void init_crivo(int n); // Inicializa o crivo de erastotenes 
int isPrimo(int num); // Retorna 1 se for primo
int palindromo(string str); // Retorna 1 se for palindromo 
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string
int proxProc(); // Retorna o id do proximo processo

int main(int argc, char* argv[]) {
	string str, palavra, frase;
	int num; //Conta os ASCII de uma palavra para verificar se é primo
	int i; //Contador
	int type; // Tipo de processamento: SMALL ou LARGE
	int length; // Otimiza o loop pára contar os ASCII de uma palavra
	int last, found; // Usado para juntar uma frase
	vector<string> pp; // palindromos primos
	vector<string> pnp; // palindromos nao primos
	vector<string> pf; // palindromos frases

	int id; // id do processador
	int tag=11; // tag para as mensagens
	MPI::Status status; // status do MPI_Recv
	char buffer[1024];

	if(argc != 3) {
		printf("Usage: ./unified type entrada.txt\n");
		exit(-1);
	}
	
	/* Indica se é para tratar o arquivo como large - palavra or small - palavra e por frase */
	type = atoi(argv[1]);

	MPI::Init(argc, argv); // Inicializa o MPI

	id = MPI::COMM_WORLD.Get_rank();

	numprocs = MPI::COMM_WORLD.Get_size();

	cout << numprocs << endl;

	// Verifica o rank
	if(id == 0) {
		atual_proc = 1;
		/* No texto maior verificamos se a palavra eh um numero primo */
		//if(type == LARGE)
		//	init_crivo(MAX_NUM); /* Inicializa o crivo de erastotenes */

		// Master node	
		ifstream entrada(argv[2], ifstream::in);
		int word_count = 0;
		int prox;
		while(!entrada.eof()) {
			if(type == SMALL) {
        		getline(entrada,str);
                found = str.find_first_of(" .!?");
                last = 0;
                while(found != string::npos) {
                	if(last != 0) {
                		last++;
                	} 
                                  
        			palavra = trim(str.substr(last, found-last));
        			if(!palavra.empty()) {
        				word_count++;
                    	frase.append(palavra);

						// Envia o tamanho da string e a string
						prox = proxProc();
						length = frase.size();
						// buffer, size, type, dest, tag
						MPI::COMM_WORLD.Send(&length, 1, MPI::INT, prox, tag);
						MPI::COMM_WORLD.Send(frase.c_str(), length, MPI::CHAR, prox, tag);
        			    /*
						if(palindromo(palavra))
        					pp.push_back(palavra);
						*/
        			}
                                                                      
        			if(str[found] != ' ') {
        				/*
						if(word_count >= 2)
        					if(palindromo(frase))
        						pf.push_back(frase);
                          */                                            
        				frase = "";
        				word_count = 0;
        			} 
        		                                                     
        			last = found;
        			found = str.find_first_of(" .!?", found+1);
        		}
        		if(last == 0) {
        			palavra = trim(str.substr(last, found-last));
        			if(!palavra.empty()) {
        				word_count++;
        				frase.append(palavra);
        				/*
						if(palindromo(palavra))
        					pp.push_back(palavra);
							*/
        			}
        		} else {
        			palavra = trim(str.substr(last+1, found-last));
        			if(!palavra.empty()) {
        				word_count++;
        				frase.append(palavra);
						/*
        				if(palindromo(palavra))
        					pp.push_back(palavra);
							*/
        			}
        		}
        	} else if (type == LARGE) {
        		entrada >> str;

				// Envia o tamanho da string e a string
                prox = proxProc();
                length = frase.size();
                // buffer, size, type, dest, tag
                MPI::COMM_WORLD.Send(&length, 1, MPI::INT, prox, tag);
                MPI::COMM_WORLD.Send(frase.c_str(), length, MPI::CHAR, prox, tag);
				/*
        		if(palindromo(str)) {
        			num=0;
        			length = str.size();
        			for(i=0; i<length; i++)
        				num += (int) str[i];
                                                                      
        			if(isPrimo(num) != 0)
        				pp.push_back(str);
        			else
        				pnp.push_back(str);
        		}
				*/
        	}
        }
	
		// Envia um tamanho vazio indicando que acabou a leitura
		for(i=1; i<numprocs; i++) {
			length = -1;
			MPI::COMM_WORLD.Send(&length, 1, MPI::INT, prox, tag);
		}
		
		// Imprime os palindromos primos e os não primos
/*
		vector<string>::iterator it;
        if(type==LARGE)
        	cout << "MASTER: Palindromos Primos (" << pp.size() << ")" << endl;
        else
        	cout << "MASTER: Palindromos (" << pp.size() << ")" << endl;
                                                                                     
        for(it = pp.begin(); it != pp.end(); ++it)
        	cout << *it << endl;
        
        if(type==LARGE) {
        	cout << endl << "MASTER: Palindromos Nao Primos (" << pnp.size() << ")" << endl;
        	for(it = pnp.begin(); it != pnp.end(); ++it)
        		cout << *it << endl;
        } else {
        	cout << endl << "MASTER: Palindromos Frasess (" << pf.size() << ")" << endl;
        	for(it = pf.begin(); it != pf.end(); ++it)
        		cout << *it << endl;
        }*/
        entrada.close();
		

	} else {
		int word_count=0;
		int size = 0;
		while(1) {
			// Worker node
			MPI::COMM_WORLD.Recv(&size, 1, MPI::INT, 0, tag, status);
			if(size == -1)
				break;
			MPI::COMM_WORLD.Recv(buffer, 1024, MPI::CHAR, 0, tag, status);
			word_count++;
		}
		cout << "WORKER " << id << " " << buffer << " " << word_count << endl;
	}

	MPI::Finalize(); // Finaliza o MPI
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
		if(str[i] != str[length-i-1]) {
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

// Retorna o id do proximo processo
int proxProc() {
	atual_proc = (atual_proc % numprocs) ;
	if(atual_proc == 0)
		atual_proc++;
	return atual_proc;
}
