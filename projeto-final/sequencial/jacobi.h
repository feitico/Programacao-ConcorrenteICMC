#include <stdio.h>
#include <stdlib.h>

#define bool int
#define FALSE 0
#define TRUE 1

float** createMatrixFloat( int j_order );
int** createMatrix( int j_order );
float* createVectorFloat( int j_order );
int* createVector( int j_order );
void printMatrix( int** matrix, int j_order );
void printVector( float* vector, int j_order );
void deleteMatrix( float** matrix, int j_order );
void deleteVector( float* vector );
void multiply( float* matrixRes, float** matrixA, float* matrixB, int size );
int jacobiMethod( float** a, float* approx, float* b, int size, int iter, float error );
