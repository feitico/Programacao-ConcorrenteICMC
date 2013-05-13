#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

#include "omp.h"

using namespace std;

//Funcoes auxiliares
int palindromo(string str); // Retorna 1 se for palindromo 
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string

/* 1 argumento eh o numero de threads */
int main(int argc, char* argv[]) {
	int n_threads = atoi(argv[1]);
	ofstream pp("palindromo.txt");
	ofstream pf("palindromo_frase.txt");
	
	omp_set_num_threads(n_threads); /* Numero de threads do nosso programa */
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int last, found;
		int word_cont;
		stringstream ss;
		string filename;
		string str, palavra, frase;

		ss << "split" << id << ".txt";
		ss >> filename;

		ifstream in(filename.c_str());
		word_cont = 0;
		while(!in.eof()) {
			getline(in,str);
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
            			pp << palavra << endl;
            	}
                                                                  
            	if(str[found] != ' ') {
            		if(word_cont >= 2)
            			if(palindromo(frase))
							pf << frase << endl;
                                                                  
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
            			pp << palavra << endl;
            	}
            } else {
            	palavra = trim(str.substr(last+1, found-last));
            	if(!palavra.empty()) {
            		word_cont++;
            		frase.append(palavra);
            		if(palindromo(palavra))
            			pp << palavra << endl;
            	}
            }
        }
		
		in.close();
		
	}

	pp.close();
	pf.close();

	return 0;
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
