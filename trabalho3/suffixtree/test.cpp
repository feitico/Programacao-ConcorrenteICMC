#include <iostream>
#include <cstring>

#include "suffix_tree.h"


int main() {
	SUFFIX_TREE* tree;
	DBL_WORD i,len;
	char str[] = "banana";

	tree = ST_CreateTree(str, strlen(str));

	ST_PrintTree(tree);

	i = ST_FindSubstring(tree, "ana", 3);

	ST_DeleteTree(tree);
	return 0;
}
