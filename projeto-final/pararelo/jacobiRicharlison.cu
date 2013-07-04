#include <stdio.h>

#define N 7

#define check(X)\
{\
    cudaError_t cerr = X;\
    if (cerr != cudaSuccess){\
        fprintf(stderr, "GPUassert:%s at line%d.\n", cudaGetErrorString(cerr), __LINE__);\
        abort();\ 
    }\ 
}\

__global__ 
void add_arrays(int *a) 
{
	a[threadIdx.x] = 21;
}
 
int main()
{
	// Setup the arrays
	int a[N] = {15, 10, 6, 0, -11, 1,0};
  
	int *ad;
	const int isize = N*sizeof(int);
 
	// print the contents of a[]
	for(int i = 0; i < N; i++)
        printf("%d ", a[i]);
 
	// Allocate and Transfer memory to the device
	cudaMalloc( (void**)&ad, isize );  
	
	check(cudaMemcpy( ad, a, isize, cudaMemcpyHostToDevice )); 
	
	// Perform the array addition
	dim3 dimBlock( N  );  
	dim3 dimGrid ( 1  );
	add_arrays<<<dimGrid, dimBlock>>>(ad);
	
	// Copy the Contents from the GPU
	check(cudaMemcpy( a, ad, isize, cudaMemcpyDeviceToHost )); 
	cudaFree( ad );
	
	// print the contents of a[]
	for(int i = 0; i < N; i++)
        printf("%d ", a[i]);
 

	return EXIT_SUCCESS;
}
