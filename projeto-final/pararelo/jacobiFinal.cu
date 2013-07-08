#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define THREADS 512   //number of threads per block
    
#define check(X) \
{ \
    cudaError_t cerr = X; \
    if (cerr != cudaSuccess){ \
        fprintf(stderr, "GPUassert:%s at line%d.\n", cudaGetErrorString(cerr), __LINE__); \
        abort(); \
    } \
}

//each block calculates a row and each thread calculates a column
__global__ void diagonalization(float *ma, float *mb, float *x, int size){
    int row = blockIdx.x;
    int temp;   //store the value in the diagonal line "row"    

    //condition to avoid error thread out of position
    if( row < size ){
        temp = ma[row*size + row];

        for(int column = threadIdx.x; column < size; column += THREADS){

            if(row!=column){
                ma[column*size + row] = ma[column*size + row]/temp;
            } else{
                ma[column*size + row] = 0;
            }
        }
        
        //awaits all threads finalize
        __syncthreads();
        mb[row] = mb[row] / temp;
        x[row] = mb[row];
    }
}


//each block calculates a row and each thread calculates a column
__global__ void jacobiMethod(float *ma, float *x, int size, float *dev_sum){
    __shared__ float cache[THREADS];
    int row = blockIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0; //store the sum of each thread 

    //to perform the sum need a vector shared
    cache[cacheIndex] = 0;
    
    //condition to avoid error thread out of position
    if( row < size ){

        for( int column = threadIdx.x; column < size; column += THREADS){
            if(row!=column){
                temp += (ma[(column * size) + row] * x[column]);
            }
        }

        cache[cacheIndex] = temp;
       
        //awaits all threads finalize
       __syncthreads();

        //performs reductions
        int i = THREADS/2;
        while(i != 0){
            if(cacheIndex < i )
                cache[cacheIndex] += cache[cacheIndex + i];

            __syncthreads();
            i /= 2;
        }
        
        //only one thread performs the final calc
        if(cacheIndex == 0){
            dev_sum[row] = cache[0];
        }
    }
}

int main(int argc, char * argv[]){
    int j_order, j_row_test;
    float j_error, j_ite_max;
    float *h_ma, *h_mb, *h_x, *h_x0; 
    float *h_ma0, h_mb0, *h_sum;
    float *dev_ma, *dev_mb, *dev_x, *dev_sum;
    int ite = 0;
    float maxDif, maxh_x, mr;
    float result = 0;

    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%f", &j_ite_max);

    /*  allocate Memory cpu */
    h_ma = (float*)malloc(sizeof(float)*j_order*j_order);
    h_ma0 = (float*)malloc(sizeof(float)*j_order);
    h_mb = (float*)malloc(sizeof(float)*j_order);
    h_x = (float*)malloc(sizeof(float)*j_order);
    h_x0 = (float*)malloc(sizeof(float)*j_order);
    h_sum = (float*)malloc(sizeof(float)*j_order);

    /* reads the values of the matrix a */
    for(int i = 0; i<j_order; i++){
        for(int j = 0; j<j_order; j++){
            scanf("%f", &h_ma[j*j_order+i]);
            if(i==j_row_test)
                h_ma0[j] = h_ma[j*j_order+i];
        }
	}
    
    /* reads the values of the matrix b */
    for(int i=0; i<j_order; i++){
        scanf("%f", &h_mb[i]);
        if(i == j_row_test)
            h_mb0 = h_mb[i];
    }
    
    /* allocate memory gpu  */
    cudaMalloc((void**)&dev_ma, j_order*j_order*sizeof(float));
    cudaMalloc((void**)&dev_mb, j_order*sizeof(float));
    cudaMalloc((void**)&dev_x, j_order*sizeof(float));
    cudaMalloc((void**)&dev_sum, j_order*sizeof(float));
    
    /* copy array from cpu to the GPU */
    cudaMemcpy(dev_ma, h_ma, j_order*j_order*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mb, h_mb, j_order*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, h_x, j_order*sizeof(float), cudaMemcpyHostToDevice);
    
    /* order de allocation and inicialization */
    /* call function device diagonalization */
    diagonalization<<<j_order,THREADS>>>(dev_ma, dev_mb, dev_x, j_order); // OK
    
    /* copy array from gpu to the cpu */
	cudaMemcpy(h_mb, dev_mb, j_order*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, dev_x, j_order*sizeof(float), cudaMemcpyDeviceToHost);
 
    mr = FLT_MAX; //variable check error
    while(ite < j_ite_max && mr > j_error){
        // --- JacobiMethod --- //
        jacobiMethod<<<j_order , THREADS>>>(dev_ma, dev_x, j_order, dev_sum);
        
        /* copy variable dev_sum from device to host h_sum  */
        cudaMemcpy(h_sum, dev_sum, j_order*sizeof(float), cudaMemcpyDeviceToHost);
        
        // --- Check Error --- //
        maxDif = maxh_x = FLT_MIN;
        for(int i = 0 ; i < j_order; i++){
            h_x0[i] = h_x[i];
            h_x[i] = (h_mb[i] - h_sum[i]);

            if(fabs(h_x[i] - h_x0[i]) > maxDif)
                maxDif = fabs(h_x[i] - h_x0[i]);

            if(fabs(h_x[i]) > maxh_x)
                maxh_x = fabs(h_x[i]);
		}

        mr = maxDif / maxh_x;
        ite++;
        cudaMemcpy(dev_x, h_x, j_order*sizeof(float), cudaMemcpyHostToDevice);
    }
    
    //calc final value
    for(int j=0; j<j_order; j++){
        result += h_ma0[j]*h_x[j];
    }
    
    //Final Result
    printf("Iterations: %d\n", ite );
    printf("RowTest: %d => [%f] =? %f \n", j_row_test, result, h_mb0);
    
    //free memory gpu
    cudaFree(dev_ma); 
    cudaFree(dev_mb); 
    cudaFree(dev_x); 
    cudaFree(dev_sum);

    //free memory cpu
    free(h_ma); 
    free(h_mb); 
    free(h_x); 
    free(h_x0); 
    free(h_ma0); 
    free(h_sum);

    return 0;

}			
			
	   
