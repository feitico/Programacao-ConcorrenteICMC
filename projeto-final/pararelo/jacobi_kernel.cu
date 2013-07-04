#include <stdio.h>
#include <stdlib.h>

#define bool int
#define FALSE 0
#define TRUE 1

__device__ float** createMatrixFloat( int j_order ){
        float** matrix;

            /* allocates memory for the matrix */
                matrix = new float*[j_order];
                    for( int i = 0; i < j_order; i++ )
                                matrix[i] = new float[j_order];

                                    return matrix;
}

__device__ int** createMatrix( int j_order ){
        int** matrix;

            /* allocates memory for the matrix*/
                matrix = new int*[j_order];
                    for(int i=0; i<j_order; i++)
                                matrix[i] = new int[j_order];

                                    return matrix;
}

__device__ float* createVectorFloat( int j_order ){
    float *matrix;

    /* allocates memory for the array */
    matrix = new float[j_order];

    return matrix;
}

__device__ int* createVector( int j_order ){
    int *matrix;

    /* allocates memory for the array */
    matrix = new int[j_order];

    return matrix;
}

__device__ void printMatrix( int** matrix, int j_order ){
     /* prints the matrix */
    for(int i = 0; i < j_order; i++ ){
        for(int j = 0; j < j_order; j++){
            printf("%d ", matrix[i][j]);
        }

        printf("\n");
    }
}

__device__ void printVector( float* vector, int j_order ){
    /* prints the vector */
    for(int i = 0; i < j_order; i++){
       printf("%f ", vector[i]);
    }
}

__device__ void deleteMatrix( float** matrix, int j_order ){
    /* deletes the matrix */
    for(int i = 0; i < j_order; i++)
        delete [] matrix[i];
    delete [] matrix;
}

__device__ void deleteVector( float* vector ){
    /* deletes the vector */
    delete vector;
}

__device__ float Abs( float x, float x0 ){
    if( (x - x0) >= 0 )
        return ( x - x0 );
    else
        return ( x0 - x );
}

__device__ float Abs( float x ){
    if( x >= 0 )
        return x;
    else
        return -x;
}

__device__ bool checkStoppingCriterion( float* x, float* x0, int n, float error ){
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

    //Mr
    if( (maxNumerator/maxDenominator) <= error )
        return TRUE;
    else
        return FALSE;
}

__device__ void multiply( float* matrixRes, float** matrixA, float* matrixB, int size )
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

__device__ void jacobiMethod( float** a, float* approx, float* b, int size, int iter, float error ){
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

		if( ctr > 0 )
            approachAchieved = checkStoppingCriterion( approx, approx0, size, error );

        lastIteration = ctr;
        ctr++;
    }

    deleteMatrix(Dinv, size);
    deleteMatrix(R, size);
    deleteVector(matrixRes );
    deleteVector(temp );
    deleteVector(approx0 );

    //return lastIteration;
}

__global__ void kernel(float** A, int size){

        printf("o valor de a e\n");

        /*for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                        printf("%f ", A[i][j]);
                }
                printf("\n");
        }
*/
}

int main() {
    int j_order;
    int j_row_test;
    float j_error;
    int j_ite_max, qtdIteracoes;
    float **ma_d_host;
    float *mb_d_host, *approx_d_host;
        float **c_ma_d;
        float *c_mb_d, *c_approx_d;
    float result = 0;

        printf("digite a ordem da matriz\n");
    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);

        /* allocate memory cpu */
        ma_d_host = (float **)calloc(j_order, sizeof(float *));
        for(int i = 0; i < j_order; i++)
                ma_d_host[i] = (float *)calloc(j_order, sizeof( float ));

        mb_d_host = (float *)calloc(j_order, sizeof(float));

		approx_d_host = (float *)calloc(j_order, sizeof(float));


        /* reads the values of the matrix a */
    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%f", &ma_d_host[i][j]);

    /* assign the values of the first approximation */
    for( int i=0; i<j_order; i++ )
        approx_d_host[i] = 0;

    /* reads the values of the matrix b */
    for(int i=0; i<j_order; i++)
        scanf("%f", &mb_d_host[i]);

    printf("ira imprimir os valores lidos da matriz A\n");
    for(int i = 0; i < j_order; i++){
            for(int j = 0; j < j_order; j++){
                    printf("%f ", ma_d_host[i][j]);
            }
            printf("\n");
    }


    printf("ira imprimir os valores lidos da matriz B\n");
    for(int i = 0; i < j_order; i++){
            printf("%f ", mb_d_host[i]);
    }

    printf("\nalocara memoria para a GPU\n");

    /* allocate memory gpu */
    //allocate for j_order float pointer
    cudaMalloc((void***)(&c_ma_d), sizeof(float *) * j_order );
    
    for( int i = 0; i < j_order; i++ ){
        float* temp;

        //allocate for j_order float in each float pointer
        cudaMalloc((void**) &(temp), sizeof(float) * j_order );

        //copy data
        cudaMemcpy(temp, ma_d_host[i], sizeof(float) * j_order, cudaMemcpyHostToDevice );

        cudaMemcpy(c_ma_d+i, &temp, sizeof(float*), cudaMemcpyHostToDevice );
    }

    kernel<<<1,1>>>(c_ma_d, j_order);

    

    printf("saiu do kernel\n");
    cudaDeviceSynchronize();
    cudaDeviceReset();
    /*qtdIteracoes = jacobiMethod( ma, approx, mb, j_order, j_ite_max, j_error );

    for( int i = 0; i < j_order; i++ )
        result += ma[j_row_test][i] * approx[i];
    
    printf("Iterations: %d\n", qtdIteracoes);
    printf("RowTest: %d => [%f] =? %f \n", j_row_test, result, mb[j_row_test]);

    deleteMatrix( ma, j_order );
    deleteVector( mb );
    deleteVector( approx );
*/
    return 0;
}
