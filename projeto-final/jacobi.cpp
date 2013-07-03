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

void printVector( float* vector, int j_order ){
    /* prints the vector */
    for(int i = 0; i < j_order; i++){
       printf("%f ", vector[i]); 
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

float Abs( float x, float x0 ){
    if( (x - x0) >= 0 )
        return ( x - x0 );
    else
        return ( x0 - x ); 
}

float Abs( float x ){
    if( x >= 0 )
        return x;
    else
        return -x;
}

bool checkStoppingCriterion( float* x, float* x0, int n, float error ){
    float* Dif;
    float maxNumerator = 0;
    float maxDenominator = 0;

    Dif = createVectorFloat( n );
   
    for(int i = 0; i < n; i++){
        Dif[i] = Abs(x[i], x0[i]);
        
        if( Dif[i] > maxNumerator )
            maxNumerator = Dif[i];
    }

    for(int i = 0; i < n; i++){
        if( Abs(x[i]) > maxDenominator) {
            maxDenominator = Abs(x[i]);
        }
    }

    printf("valor de approach %f\n", maxNumerator/maxDenominator);
    //Mr
    if( (maxNumerator/maxDenominator) <= error )
        return TRUE;
    else
        return FALSE;
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

void jacobiMethod( float** a, float* approx, float* b, int size, int iter, float error ){
    float** Dinv; 
    float**R;
    float* matrixRes; 
    float* temp;
    float* approx0;
    int ctr = 0, octr;
    int lastIteration = 0; 
    bool approachAchieved = 0;

    Dinv = createMatrixFloat( size );
    R = createMatrixFloat( size );
    matrixRes = createVectorFloat( size );
    temp = createVectorFloat( size );
    approx0 = createVectorFloat( size );
    
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

    while( ctr <= iter && approachAchieved == FALSE){
        //copy values of approx to approx0 
        for(int i = 0; i < size; i++)
            approx0[i] = approx[i];
        
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
        
        printVector( approx, size );

        if( ctr > 0 )
            approachAchieved = checkStoppingCriterion( approx, approx0, size, error );
        
        lastIteration = ctr;
        ctr++;
    }

    printf("The Value after iteration %d is\n", lastIteration);
    for( int row = 0; row < size; row++ ){
        //display the value after the pass
        printf("%.3f\n", approx[row]);
    }
}
