#include <stdio.h>
#include <stdlib.h>

#define bool int
#define FALSE 0
#define TRUE 1

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
    /*
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
    */
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

    /*Dinv = createMatrixFloat( size );
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
    */
}

__global__ void kernel( float* B ){
    /**
    * blockIdx.x contém o Id do bloco a ser executado
    */
    //int tid = blockIdx.x;    // this thread handles the data at its thread id;
    //    B[tid] = 5.4;
    B[0] = 4.3;
}

int main() {
    int j_order;
    int j_row_test;
    float j_error;
    int j_ite_max, qtdIteracoes;
    float **h_ma, **d_ma;
    float *h_mb, *h_approx;
    float *d_mb, *d_approx;
    float result = 0;

    printf("digite a ordem da matriz\n");
    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);

    /* allocate memory cpu */
    h_ma = (float **)malloc(j_order * sizeof(float *));
    for(int i = 0; i < j_order; i++)
            h_ma[i] = (float *)malloc(j_order * sizeof( float ));

    h_mb = (float *)malloc(j_order * sizeof(float));

    h_approx = (float *)malloc(j_order * sizeof(float));

    /* reads the values of the matrix a */
    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%f", &h_ma[i][j]);

    /* assign the values of the first approximation */
    for( int i=0; i<j_order; i++ )
        h_approx[i] = 0;

    /* reads the values of the matrix b */
    for(int i=0; i<j_order; i++)
        scanf("%f", &h_mb[i]);

    printf("ira imprimir os valores lidos da matriz A\n");
    for(int i = 0; i < j_order; i++){
        for(int j = 0; j < j_order; j++){
            printf("%f ", h_ma[i][j]);
        }
        printf("\n");
    }


    printf("ira imprimir os valores lidos do vetor B\n");
    for(int i = 0; i < j_order; i++){
        printf("%f ", h_mb[i]);
    }

    printf("\nalocara memoria para a GPU\n");

    /* allocate memory gpu to matrix */

    //allocate for j_order float pointer
    /*cudaMalloc((void***)(&c_ma_d), sizeof(float *) * j_order );
    
    for( int i = 0; i < j_order; i++ ){
        float* temp;

        //allocate for j_order float in each float pointer
        cudaMalloc((void**) &(temp), sizeof(float) * j_order );

        //copy data
        cudaMemcpy(temp, ma_d_host[i], sizeof(float) * j_order, cudaMemcpyHostToDevice );

        cudaMemcpy(c_ma_d+i, &temp, sizeof(float*), cudaMemcpyHostToDevice );
    }*/

    /* allocate memory gpu to vector */
    cudaMalloc( (void**)&d_mb, j_order * sizeof(float));    
    
    /* copy the arrays mb_d_host to the GPU */
    cudaMemcpy( d_mb, h_mb, j_order * sizeof(float), cudaMemcpyHostToDevice );
    
    // Perform the array
    dim3 dimBlock( j_order );
    dim3 dimGrid( 1 );
    kernel<<<1,1>>>( d_mb );

    printf("saiu do kernel\n");
    
    //copy the array c_mb_d from GPU to the CPU
    cudaMemcpy(h_mb, d_mb, j_order * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mb);

    printf("imprimira os valores do vetor B apos sair do kernel\n");
    for(int i = 0; i < j_order; i++)
        printf("%f ", h_mb[i]);

    printf("\nimprimira os valores da matriz A apos sair do kernel\n");
    for(int i = 0; i < j_order; i++){
        for(int j = 0; j < j_order; j++)
            printf("%f ", h_ma[i][j]);

        printf("\n");
    }

    //cudaDeviceSynchronize();
    //cudaDeviceReset();
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
