#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

using namespace std;

//Funcao auxiliar
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string

/* 
 * 1 argumento - numero de particoes de arquivo
 * 2 argumento - arquivo texto a ser particionado
 */
int main(int argc, char *argv[]) {
	int n = atoi(argv[1]);
	ifstream in(argv[2]);
	ofstream out[n];
	int i, last, found;
	string str, palavra, frase;

	/* Cria os arquivos particionados */
	for(i=0; i<n; i++) {
		stringstream ss;
		string filename;
		ss << "split" << i << ".txt";
		ss >> filename;
		out[i].open(filename.c_str());
	}
	
	/* Percorro o arquivo de entrada e particiono no arquivo de saida, uma frase por arquivo */
	i=0;
	while(!in.eof()) {
		getline(in,str);                                      		
        found = str.find_first_of(" .!?");
        last = 0;
        while(found != string::npos) {
        	if(last != 0) {
        		last++;
        	} 
                          
        	palavra = trim(str.substr(last, found-last));
        	if(!palavra.empty()) 
            	frase.append(palavra);
                                                              
        	if(str[found] != ' ') {
        		out[i] << frase << "." << endl;
                                                              
        		frase = "";
				i = (i+1) % n;
        	} 
                                                             
        	last = found;
        	found = str.find_first_of(" .!?", found+1);
        }
        if(last == 0) {
        	palavra = trim(str.substr(last, found-last));
        	if(!palavra.empty()) {
        		frase.append(palavra);
        	}
        } else {
        	palavra = trim(str.substr(last+1, found-last));
        	if(!palavra.empty()) {
        		frase.append(palavra);
        	}
        }
	}

	/* Fecha os arquivos */
	in.close();
	
	for(i=0; i<n; i++) {
		out[i].close();
	}

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
