#include "patricia.h"

int main() {
	char* s;
	patricia* tree=new patricia;
	s = (char*)malloc(sizeof(char)*12); strcpy(s,"romane");     tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"romanus");        tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"romulus"); tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"rubens");      tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"ruber");    tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"rubicon");      tree->add(s,NULL); assert(0==tree->validate());
    s = (char*)malloc(sizeof(char)*12); strcpy(s,"rubicundus");   tree->add(s,NULL); assert(0==tree->validate());


	tree->print();

	return 0;
}
