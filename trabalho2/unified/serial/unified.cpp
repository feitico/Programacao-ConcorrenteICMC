#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>

#define MAX_NUM 8000
#define RAIZ_MAX_NUM 90
#define SMALL 0
#define LARGE 1

using namespace std;

int primo[MAX_NUM];

// Funcoes auxiliares
void init_crivo(int n); // Inicializa o crivo de erastotenes 
int isPrimo(int num); // Retorna 1 se for primo
int palindromo(string str); // Retorna 1 se for palindromo 
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string

int main(int argc, char* argv[]) {
	string str, palavra, frase;
	int num; //Conta os ASCII de uma palavra para verificar se é primo
	int i; //Contador
	int type; // Tipo de processamento: SMALL ou LARGE
	int size; // Otimiza o loop pára contar os ASCII de uma palavra
	int last, found; // Usado para juntar uma frase
	int word_cont; // Conta quantas palavras possue uma frase, frase >= 2 palavras
	ifstream entrada(argv[2], ifstream::in);
	vector<string> pp; // palindromos primos
	vector<string> pnp; // palindromos nao primos
	vector<string> pf; // palindromos frases

	if(argc != 3) {
		printf("Usage: ./unified type entrada.txt\n");
		exit(-1);
	}
	
	/* Indica se é para tratar o arquivo como large - palavra or small - palavra e por frase */
	type = atoi(argv[1]); 

	/* No texto maior verificamos se a palavra eh um numero primo */
	if(type == LARGE)
		init_crivo(MAX_NUM); /* Inicializa o crivo de erastotenes */

	word_cont = 0;
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
					word_cont++;
	            	frase.append(palavra);
				    if(palindromo(palavra))
						pp.push_back(palavra);
				}

				if(str[found] != ' ') {
					if(word_cont >= 2)
						if(palindromo(frase))
							pf.push_back(frase);

					frase = "";
					word_cont = 0;
				} 
			                                                     
				last = found;
				found = str.find_first_of(" .!?", found+1);
			}
			if(last == 0) {
				palavra = trim(str.substr(last, found-last));
				if(!palavra.empty()) {
					word_cont++;
					frase.append(palavra);
					if(palindromo(palavra))
						pp.push_back(palavra);
				}
			} else {
				palavra = trim(str.substr(last+1, found-last));
				if(!palavra.empty()) {
					word_cont++;
					frase.append(palavra);
					if(palindromo(palavra))
						pp.push_back(palavra);
				}
			}
		} else if (type == LARGE) {
			entrada >> str;
			if(palindromo(str)) {
				num=0;
				size = str.size();
				for(i=0; i<size; i++)
					num += (int) str[i];

				if(isPrimo(num) != 0)
					pp.push_back(str);
				else
					pnp.push_back(str);
			}
		}
	}

	// Imprime os palindromos primos e os não primos
	vector<string>::iterator it;
	if(type==LARGE)
		cout << "Palindromos Primos (" << pp.size() << ")" << endl;
	else
		cout << "Palindromos (" << pp.size() << ")" << endl;

	for(it = pp.begin(); it != pp.end(); ++it)
		cout << *it << endl;
	
	if(type==LARGE) {
		cout << endl << "Palindromos Nao Primos (" << pnp.size() << ")" << endl;
		for(it = pnp.begin(); it != pnp.end(); ++it)
			cout << *it << endl;
	} else {
		cout << endl << "Palindromos Frases (" << pf.size() << ")" << endl;
		for(it = pf.begin(); it != pf.end(); ++it)
			cout << *it << endl;
	}
	
	entrada.close();

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
