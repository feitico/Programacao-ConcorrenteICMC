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
        if(strcmp(words[imid], word) < 0)
            imin = imid + 1;
        else if (strcmp(words[imid], word) > 0)
            imax = imid - 1;
        else
            return imid;
    }
    return -1;
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
    for(int i=0; i<qtd; i++)
         words[i] = (char*) malloc(sizeof(char) * maxlength);
}

void Dict::insert(int pos, char* word) {
    if(pos < qtd) {
        strcpy(words[pos], word);
        marked[pos] = 0;
    }
}

int Dict::markWord(char* word) {
    int indice = binary_search(words, word, qtd);
    if(indice != -1) {
        if(marked[indice] != 0) {
            marked[indice] = 1;
            qtdMarked++;
            return indice;
        } else
            return -1;
    } else 
        return -1;
}

void Dict::print()
{
    for(int i=0; i<qtd; i++)
        cout << words[i] << endl;
}

char** Dict::getWords()
{
    return words;
}

int* Dict::getMarked()
{
    return marked;
}

int Dict::getQtd() 
{
    return qtd;
}

int Dict::getQtdMarked() {
    return qtdMarked;
}
