#include <stdio.h>
#include <stdlib.h>

int** createMatrix( int j_order );
int* createVector( int j_order );
void printMatrix( int** matrix, int j_order );
void printVector( int* vector, int j_order );
void deleteMatrix( int** matrix, int j_order );
void deleteVector( int* vector );
