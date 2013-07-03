#include "jacobi.h"

float** createMatrixFloat( int j_order ){
    float** matrix;

    /* allocates memory for the matrix */
    matrix = new float*[j_order];
    for( int i = 0; i < j_order; i++ )
        matrix[i] = new float[j_order];

    return matrix;
}

int** createMatrix( int j_order ){
    int** matrix;

    /* allocates memory for the matrix*/
    matrix = new int*[j_order];
    for(int i=0; i<j_order; i++)
        matrix[i] = new int[j_order];

    return matrix;
}

float* createVectorFloat( int j_order ){
    float *matrix;

    /* allocates memory for the array */
    matrix = new float[j_order];

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

void deleteMatrix( float** matrix, int j_order ){
    /* deletes the matrix */
    for(int i = 0; i < j_order; i++)
        delete [] matrix[i];
    delete [] matrix;   
}

void deleteVector( float* vector ){
    /* deletes the vector */
    delete vector;
}

void multiply( float* matrixRes, float** matrixA, float* matrixB, int size )
{  
    //function to perform multiplication
    for( int ctr = 0; ctr < size; ctr++ )
    {
        matrixRes[ctr] = 0;
        for(int navigate = 0; navigate < size; navigate ++){
            matrixRes[ctr] = matrixRes[ctr]+matrixA[ctr][navigate]
                            *matrixB[navigate];
        }
    }
}

void jacobiMethod( float** a, float* approx, float* b, int size, int iter ){
    float** Dinv; 
    float**R;
    float* matrixRes; 
    float* temp;
    int ctr = 1, octr;
    
    
    Dinv = createMatrixFloat( size );
    R = createMatrixFloat( size );
    matrixRes = createVectorFloat( size );
    temp = createVectorFloat( size );

    //We calculate the diagonal inverse matrix make all other entries
    //as zero except Diagonal entries whose resciprocal we store
    for(int row = 0; row < size; row++){
        for( int column = 0; column < size; column++ ){
            if( row == column )
                Dinv[row][column] = 1/a[row][column];
            else
                Dinv[row][column] = 0;
        }
    }

    for(int row = 0; row < size; row++){
        //calculating the R matrix L+U
        for( int column = 0; column < size; column++ ){
            if( row == column ){
                R[row][column] = 0;
            }
            else if( row != column ){
                R[row][column] = a[row][column];
            }
        }
    }

    while( ctr <= iter ){
        //multiply L+U and the approximation
        multiply( matrixRes, R, approx, size );
        
        for( int row = 0; row < size; row++ ){
            //the matrix( b-Rx )
            temp[row] = b[row] - matrixRes[row]; //the matrix( b-Rx ) i
        }
        
        //multiply D inverse and ( b-Rx )
        multiply( matrixRes, Dinv, temp, size );

        for( octr = 0; octr < size; octr++ ){
            //store matrixRes value int the nex approximation
            approx[octr] = matrixRes[octr];
        }

        printf("The Value after iteration %d is\n", ctr);
        for( int row = 0; row < size; row++ ){
            //display the value after the pass
            printf("%.3f\n", approx[row]);
        }

        ctr++;
    }
}
