#include <stdio.h>
#include <stdlib.h>


/**
 * Hello World in CUDA
 * HOST - memória e processador da CPU local
 * Device - memória e processador da placa gráfica
 * Kernel - função que executa no device
 */


/**
 * __global__ - alerta o compilador que a função deve ser compilada para ser executada no device
 */
__global__ void kernel(void){}

int main( void ) {
	
	kernel<<<1,1>>>();
	printf( "Hello, World!\n" );
	return 0; 
}
