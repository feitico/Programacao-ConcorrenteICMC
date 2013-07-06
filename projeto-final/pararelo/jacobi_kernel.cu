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

#define THREADS 256

 //We calculate the diagonal inverse matrix make all other entries
 //as zero except Diagonal entries whose resciprocal we store
__global__ void diagonalization(float* a, float *Dinv, float *R, int size, float *approx, float *approx0){
    
    int row = blockIdx.x;

    if( row < size ){

        for( int column = threadIdx.x; column < size; column+=THREADS ){
            if( row == column )
                Dinv[row*size + column] = 1/a[row*size + column];
            else
                Dinv[row*size + column] = 0;
        }
    
        //calculating the R matrix L+U
        for( int column = threadIdx.x; column < size; column+=THREADS ){
            if( row == column ){
                R[row*size + column] = 0;
            }
            else if( row != column ){
                R[row*size + column] = a[row*size + column];
            }
        }
        
        __syncthreads();
        //copy values of approx to approx0 
        approx0[row] = approx[row];
    } 
}

__global__ void jacobiMethod( float* a, float* Dinv, float* R, float* approx, float* b, float* matrixRes, float* temp, int size, int iter ){
    __shared__ float cache[THREADS];
    int row = blockIdx.x;
    int cacheIndex = threadIdx.x;
    float temp1 = 0;

    cache[cacheIndex] = 0;

    //multiply L+U and the approximation
    //function to perform multiplication
    if( row < size )
    {
        temp1 = 0;
        //matrixRes[row] = 0;
        for(int column = threadIdx.x; column < size; column+=THREADS){
            temp1 += R[row*size + column]
                            *approx[column];
        }
    
        cache[cacheIndex] = temp1;
        __syncthreads();

        int i = THREADS/2;
        while( i != 0){
            if(cacheIndex < i)
                cache[cacheIndex] += cache[cacheIndex + i];

            __syncthreads();
            i /= 2;
        }

        if(cacheIndex == 0){
            matrixRes[blockIdx.x] = cache[0];        
        }

        //the matrix( b-Rx )
        temp[row] = b[row] - matrixRes[row]; //the matrix( b-Rx ) i
        
        //multiply D inverse and ( b-Rx )
        //function to perform multiplication
        //matrixRes[row] = 0;
        temp1 = 0;
        for(int column = threadIdx.x; column < size; column+=THREADS){
            temp1 += Dinv[row*size + column]
                            *temp[column];
        }
        
        cache[cacheIndex] = temp1;
        __syncthreads();

        i = THREADS/2;
        while( i != 0){
            if(cacheIndex < i)
                cache[cacheIndex] += cache[cacheIndex + i];

            __syncthreads();
            i /= 2;
        }

        if(cacheIndex == 0){
            matrixRes[blockIdx.x] = cache[0];        
        }

        //store matrixRes value int the next approximation
        approx[row] = matrixRes[row];
    }
}

int main() {
    int j_order, j_row_test;
    int j_ite_max, ctr = 0;
    int qtdIterations = 0;
    float j_error, maxNumerator = 0, maxDenominator = 0, absApprox = 0;
    float *h_ma, *h_mb, *h_approx, *h_Dif, *h_approx0;
    float *d_ma, *d_mb, *d_approx;
    float *d_Dinv, *d_R;
    float *d_matrixRes, *d_temp, *d_approx0; 
    bool approachAchieved = 0;

    float result = 0;

    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);

    /* allocate memory cpu */
    h_ma = (float *)malloc(j_order * j_order * sizeof(float));
    h_mb = (float *)malloc(j_order * sizeof(float));
    h_approx = (float *)malloc(j_order * sizeof(float));
    h_Dif = (float *)malloc(j_order * sizeof(float));
    h_approx0 = (float *)malloc(j_order * sizeof(float));

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
    
    /* copy the arrays to the GPU */ 
    check(cudaMemcpy( d_ma, h_ma, j_order * j_order * sizeof(float), cudaMemcpyHostToDevice ));
    check(cudaMemcpy( d_mb, h_mb, j_order * sizeof(float), cudaMemcpyHostToDevice ));
    check(cudaMemcpy( d_approx, h_approx, j_order * sizeof(float), cudaMemcpyHostToDevice));

    /* initialization time */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mSec = 0;

    /* order allocation and initialization */
    diagonalization<<<j_order, THREADS>>>( d_ma, d_Dinv, d_R, j_order, d_approx, d_approx0);
    check(cudaMemcpy(h_approx0, d_approx0, j_order * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&mSec, start, stop);
    printf("time diagonalization %f\n", mSec);
    
    cudaEventRecord(start);

    while( ctr <= j_ite_max && approachAchieved == FALSE  ){
        
        // --- jacobiMethod --- //
        jacobiMethod<<<j_order,THREADS>>>( d_ma, d_Dinv, d_R, d_approx, d_mb, d_matrixRes, d_temp, j_order, j_ite_max );
 
        check(cudaMemcpy(h_approx, d_approx, j_order * sizeof(float), cudaMemcpyDeviceToHost));

        //--- Check Error --- //
     	if( ctr > 0 ){
            //fara a verificacao do erro
            maxNumerator = 0;
            maxDenominator = 0;
            for(int i = 0; i < j_order; i++){
                if((h_approx[i] - h_approx0[i]) >= 0 )
                    h_Dif[i] = h_approx[i] - h_approx0[i];
                else
                    h_Dif[i] = h_approx0[i] - h_approx[i];
                //------------------------------------

                if( h_Dif[i] > maxNumerator )
                    maxNumerator = h_Dif[i];
            }

            for(int i = 0; i < j_order; i++){
                //Abs(float x)
                if( h_approx[i] >= 0 )
                    absApprox = h_approx[i];
                else
                    absApprox = -h_approx[i];
                //----------------------
                if( absApprox > maxDenominator) {
                    maxDenominator = absApprox;
                }
            }

            //Mr
            if( (maxNumerator/maxDenominator) <= j_error )
                approachAchieved = TRUE;
            else
                approachAchieved = FALSE;
        }

        //copy values of approx to approx0 
        for(int i = 0; i < j_order; i++)
            h_approx0[i] = h_approx[i];

        qtdIterations = ctr;
        ctr++;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&mSec, start, stop);
    printf("Method jacobi: %f\n", mSec);

    cudaFree(d_ma);
    cudaFree(d_Dinv);
    cudaFree(d_R);

    cudaFree(d_approx);
    cudaFree(d_mb);
    cudaFree(d_matrixRes);
    cudaFree(d_temp);
    cudaFree(d_approx0);

    //calculate result
    for( int i = 0; i < j_order; i++ )
        result += h_ma[j_row_test*j_order + i] * h_approx[i];

    printf("Iterations: %d\n", qtdIterations );
    printf("RowTest: %d => [%f] =? %f \n", j_row_test, result, h_mb[j_row_test]);

    cudaThreadSynchronize();
    
    free(h_ma);
    free(h_mb);
    free(h_approx);
    free(h_approx0);
    free(h_Dif);
    return 0;
}
