#include <stdio.h>
#include <stdlib.h>

/**
 * Função que realiza a soma de a + b e retorna um inteiro.
 * __device__ apenas o dispositivo pode chamar esta função.
 */
__device__ int addem( int a, int b ) {
    return a + b;
}

/**
 * Função que vai ser chamada do host para o dispositivo.
 * realiza a soma de a + b e guarda o resultado no ponteiro passado
 */
__global__ void add( int a, int b, int *c ) {
    *c = addem( a, b );
}

int main(void){
    /** variavel de resposta **/	
    int c;
    /** ponteiro a ser passado para as funções no dispositivo **/
    int *dev_c;
    /** alocação de memória no dispositivo **/
    cudaMalloc((void**)&dev_c, sizeof(int));
    /** chamada a função add para ser executada no dispositivo **/
    add<<<1,1>>>(2,7,dev_c);
    /** cópia do conteudo do ponteiro dev_c para a variavel c **/
    cudaMemcpy(&c,dev_c, sizeof(int),cudaMemcpyDeviceToHost);
    printf( "2 + 7 = %d\n", c );
    /** liberação de memoria alocada no dispositivo **/
    cudaFree(dev_c);

    return 0;
}
