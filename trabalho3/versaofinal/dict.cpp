#include <cstring>
#include <cstdlib>
#include <iostream>

#include "dict.hpp"

using namespace std;

int binary_search(char** words, const char* word, int qtd)
{
    int imin = 0;
    int imax = qtd-1;
                                                            
    // Continua buscando enquanto [imin, imax] nao eh vazio
    while(imax >= imin) {
        int imid = (imin + imax) / 2;
                                                            
        // Decide em qual sub array para procurar
        if(strlen(words[imid]) == 0)
                imax = imid - 1;
        else {
            if(strcmp(words[imid], word) < 0)
                imin = imid + 1;
            else if (strcmp(words[imid], word) > 0)
                imax = imid - 1;
            else
                return imid;
        }
    }
    return NOT_FOUND;
}

Dict::Dict()
{
}

Dict::~Dict()
{
    for(int i=0; i<qtd; i++)
        free(words[i]);
    free(words);
    free(marked);
}

void Dict::init(int qtd, int maxlength)
{
    this->qtd = qtd;
    this->maxlength = maxlength;
    qtdMarked = 0;
    marked = (int*) malloc(sizeof(int) * qtd);
    words = (char**) malloc(sizeof(char*) * qtd);
    qtdMarkedLength = (int*) malloc(sizeof(int) * qtd);
    for(int i=0; i<qtd; i++) {
         words[i] = (char*) malloc(sizeof(char) * maxlength);
         marked[i] = 0;
         qtdMarkedLength[i] = 0;
    }
}

void Dict::insert(int pos, char* word) {
    if(pos < qtd) {
        strcpy(words[pos], word);
        marked[pos] = 0;
    }
}

int Dict::markWord(char* word) {
    int idx = binary_search(words, word, qtd);
    if(idx != NOT_FOUND) {
        if(marked[idx] == 0) {
            marked[idx] = 1;
            qtdMarked++;
            qtdMarkedLength[strlen(words[idx])]++;
            return idx;
        } else
            return NOT_FOUND;
    } else 
        return NOT_FOUND;
}

int Dict::search(char* word) {
    return binary_search(words, word, qtd);
}

int Dict::markPos(int idx) {
	if(idx < qtd) {
		if(marked[idx] == 0) {
        	marked[idx] = 1;
			qtdMarked++;
            qtdMarkedLength[strlen(words[idx])]++;
    	    return 1;
	    } else {
    	    return 0;
	    }
	} else 
		return 0;
}

void Dict::print(int ismarked)
{
    for(int i=0; i<qtd; i++) {
        if(marked[i] == ismarked)
            cout << words[i] << endl;
    }
}

char** Dict::getWords()
{
    return words;
}

int* Dict::getMarked()
{
    return marked;
}
void Dict::setMarked(int* newMarked)
{
    qtdMarked=0;
    for(int i=0; i<qtd; i++) {
        marked[i] = newMarked[i];
        if(newMarked[i])
            qtdMarked++;
    }
}

int Dict::getQtd() 
{
    return qtd;
}

int Dict::getQtdMarked() 
{
    return qtdMarked;
}

int Dict::getQtdMarkedLength(int length) 
{
    return qtdMarkedLength[length];
}

int Dict::getMaxWordLength() 
{
    return maxlength;
}
