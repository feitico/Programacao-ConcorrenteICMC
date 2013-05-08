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

int main(int argc, char* argv[]) {
	string str, frase;
	int num, i, max=0, large, type, size, last, found;
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

	while(!entrada.eof()) {
		if(type == SMALL) {
			entrada >> str;

			if(palindromo(str)) {
				num=0;
				size=str.size();
				for(i=0; i<size; i++)
					num += (int) str[i];

				pp.push_back(str);
			}
			found = str.find_first_of(".!?\n");
            last = 0;
            while(found != string::npos) {
            	frase.append(str.substr(last, found));
				
				if(palindromo(frase)) {
                	num=0;
                	size=str.size();
                	for(i=0; i<size; i++)
                		num += (int) str[i];
                                             
                	pf.push_back(str);
                }
            	
				frase = "";
                                                             
            	last = found;
            	found = str.find_first_of(".!?\n", found+1);
            }
            if(last != 0)
            	frase.append(str);

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
		cout << endl << "Palindromos Frasess (" << pf.size() << ")" << endl;
		for(it = pf.begin(); it != pf.end(); ++it)
			cout << *it << endl;
	}
	
	entrada.close();

	return 0;
}
