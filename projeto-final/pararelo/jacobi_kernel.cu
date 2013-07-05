#include <stdio.h>
#include <stdlib.h>

#define bool int
#define FALSE 0
#define TRUE 1

#define check(X) \
{ \
    cudaError_t cerr = X; \
    if (cerr != cudaSuccess){ \
        fprintf(stderr, "GPUassert:%s at line%d.\n", cudaGetErrorString(cerr), __LINE__); \
        abort(); \
    } \
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

__global__ void jacobiMethod( float* a, float* Dinv, float* R, float* approx, float* b, float* matrixRes, float* temp, float* approx0, int size, int iter, float error, int* qtdIterations, float* d_Dif ){

    int ctr = 0, octr;
    bool approachAchieved = 0;
    float maxNumerator = 0, maxDenominator = 0;

    *qtdIterations = 0;

    //We calculate the diagonal inverse matrix make all other entries
    //as zero except Diagonal entries whose resciprocal we store
    for(int row = 0; row < size; row++){
        for( int column = 0; column < size; column++ ){
            if( row == column )
                Dinv[row*size + column] = 1/a[row*size + column];
            else
                Dinv[row*size + column] = 0;
        }
    }
	
	for(int row = 0; row < size; row++){
        //calculating the R matrix L+U
        for( int column = 0; column < size; column++ ){
            if( row == column ){
                R[row*size + column] = 0;
            }
            else if( row != column ){
                R[row*size + column] = a[row*size + column];
            }
        }
    }

    while( ctr <= iter && approachAchieved == FALSE){
        //copy values of approx to approx0 
        for(int i = 0; i < size; i++)
            approx0[i] = approx[i];

        //multiply L+U and the approximation
        //function to perform multiplication
        for( int i = 0; i < size; i++ )
        {
            matrixRes[i] = 0;
            for(int navigate = 0; navigate < size; navigate++){
                matrixRes[i] = matrixRes[i]+R[i*size + navigate]
                                *approx[navigate];
            }
        }
        //-----------------------------------------------

        for( int row = 0; row < size; row++ ){
            //the matrix( b-Rx )
            temp[row] = b[row] - matrixRes[row]; //the matrix( b-Rx ) i
        }

        //multiply D inverse and ( b-Rx )
        //function to perform multiplication
        for( int i = 0; i < size; i++ )
        {
            matrixRes[i] = 0;
            for(int navigate = 0; navigate < size; navigate ++){
                matrixRes[i] = matrixRes[i]+Dinv[i*size + navigate]
                                *temp[navigate];
            }
        }

        //---------------------------------------------------

        for( octr = 0; octr < size; octr++ ){
            //store matrixRes value int the nex approximation
            approx[octr] = matrixRes[octr];
        }

		if( ctr > 0 ){
            //fara a verificacao do erro
            maxNumerator = 0;
            maxDenominator = 0;
            for(int i = 0; i < size; i++){
                d_Dif[i] = Abs(approx[i], approx0[i]);

                if( d_Dif[i] > maxNumerator )
                    maxNumerator = d_Dif[i];
            }

            for(int i = 0; i < size; i++){
                if( Abs(approx[i]) > maxDenominator) {
                    maxDenominator = Abs(approx[i]);
                }
            }

            //Mr
            if( (maxNumerator/maxDenominator) <= error )
                approachAchieved = TRUE;
            else
                approachAchieved = FALSE;
        }
        
        *qtdIterations = ctr;
        ctr++;
    }
}

int main() {
    int j_order, j_row_test;
    int j_ite_max;
    int h_qtd_it, *d_qtd_it;
    float j_error;
    float *h_ma, *h_mb, *h_approx;
    float *d_ma, *d_mb, *d_approx;
    float *d_Dinv, *d_R;
    float *d_matrixRes, *d_temp, *d_approx0, *d_Dif; 
    
    float result = 0;

    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);

    /* allocate memory cpu */
    h_ma = (float *)malloc(j_order * j_order * sizeof(float));
    h_mb = (float *)malloc(j_order * sizeof(float));
    h_approx = (float *)malloc(j_order * sizeof(float));

    /* reads the values of the matrix a */
    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%f", &h_ma[i*j_order + j]);

    /* assign the values of the first approximation */
    for( int i=0; i<j_order; i++ )
        h_approx[i] = 0;

    /* reads the values of the matrix b */
    for(int i=0; i<j_order; i++)
        scanf("%f", &h_mb[i]);

    /* allocate memory gpu */
    check(cudaMalloc( (void**)&d_ma, j_order * j_order * sizeof(float)));   
    check(cudaMalloc( (void**)&d_Dinv, j_order * j_order * sizeof(float)));
    check(cudaMalloc( (void**)&d_R, j_order * j_order * sizeof(float)));

    check(cudaMalloc( (void**)&d_approx, j_order * sizeof(float)));
    check(cudaMalloc( (void**)&d_mb, j_order * sizeof(float)));    
    check(cudaMalloc( (void**)&d_matrixRes, j_order * sizeof(float))); 
    check(cudaMalloc( (void**)&d_temp, j_order * sizeof(float)));
    check(cudaMalloc( (void**)&d_approx0, j_order * sizeof(float)));
    check(cudaMalloc( (void**)&d_Dif, j_order * sizeof(float)));
    check(cudaMalloc( (void**)&d_qtd_it, sizeof(int)));
    
    /* copy the arrays to the GPU */ 
    check(cudaMemcpy( d_ma, h_ma, j_order * j_order * sizeof(float), cudaMemcpyHostToDevice ));
    check(cudaMemcpy( d_mb, h_mb, j_order * sizeof(float), cudaMemcpyHostToDevice ));
    check(cudaMemcpy( d_approx, h_approx, j_order * sizeof(float), cudaMemcpyHostToDevice));

    // Perform the array
    //dim3 dimBlock( j_order );
    //dim3 dimGrid( 1 );

    jacobiMethod<<<1,1>>>( d_ma, d_Dinv, d_R, d_approx, d_mb, d_matrixRes, d_temp, d_approx0, j_order, j_ite_max, j_error, d_qtd_it, d_Dif );

    //copy the array d_approx from GPU to the CPU
    check(cudaMemcpy(h_approx, d_approx, j_order * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&h_qtd_it, d_qtd_it, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_ma);
    cudaFree(d_Dinv);
    cudaFree(d_R);

    cudaFree(d_approx);
    cudaFree(d_mb);
    cudaFree(d_matrixRes);
    cudaFree(d_temp);
    cudaFree(d_approx0);
    cudaFree(d_Dif);
    cudaFree(d_qtd_it);

    //calculate result
    for( int i = 0; i < j_order; i++ )
        result += h_ma[j_row_test*j_order + i] * h_approx[i];

    printf("Iterations: %d\n", h_qtd_it );
    printf("RowTest: %d => [%f] =? %f \n", j_row_test, result, h_mb[j_row_test]);

    cudaThreadSynchronize();
    
    free(h_ma);
    free(h_mb);
    free(h_approx);
    return 0;
}
