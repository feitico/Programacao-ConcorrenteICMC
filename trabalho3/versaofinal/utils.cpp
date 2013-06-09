#include <ctime>

#include "utils.h"

using namespace std;

clock_t begin; //Utilizado pelas funcoes tic() e toc() para calcular marcar o tempo inicial

// Marca o tempo de inicio
void tic() {
	begin = clock();
}

// Marca o tempo de final e calcula o tempo entre o tic e toc em milisegundos
double toc() {
	clock_t end = clock();
	return double(end - begin) / CLOCKS_PER_SEC;
}
