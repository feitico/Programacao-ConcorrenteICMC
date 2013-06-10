#include <ctime>
#include <mpi.h>
#include "utils.h"

using namespace std;

double begin; //Utilizado pelas funcoes tic() e toc() para calcular marcar o tempo inicial

// Marca o tempo de inicio
void tic() {
	begin = MPI::Wtime();
}

// Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
double toc() {
	double end = MPI::Wtime();
	return end - begin;
}












































































