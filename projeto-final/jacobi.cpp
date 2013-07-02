#include "jacobi.h"

int** createMatrix( int j_order ){
    int** matrix;

    /* allocates memory for the matrix*/
    matrix = new int*[j_order];
    for(int i=0; i<j_order; i++)
        matrix[i] = new int[j_order];

    return matrix;
}

int* createVector( int j_order ){
    int *matrix;

    /* allocates memory for the array */
    matrix = new int[j_order];

    return matrix;
}

void printMatrix( int** matrix, int j_order ){
     /* prints the matrix */
    for(int i = 0; i < j_order; i++ ){
        for(int j = 0; j < j_order; j++){
            printf("%d ", matrix[i][j]);
        }
        
        printf("\n");
    }    
}

void printVector( int* vector, int j_order ){
    /* prints the vector */
    for(int i = 0; i < j_order; i++){
       printf("%d ", vector[i]); 
    }    
}

void deleteMatrix( int** matrix, int j_order ){
    /* deletes the matrix */
    for(int i = 0; i < j_order; i++)
        delete [] matrix[i];
    delete [] matrix;   
}

void deleteVector( int* vector ){
    /* deletes the vector */
    delete vector;
}
