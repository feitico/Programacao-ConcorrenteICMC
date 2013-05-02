#include <stdio.h>

#include "omp.h"

void pooh(int id, double* A) {
	printf("A[%d] = %f ", id, A[id]);
}

int main() {
	double A[4] = {4,3,2,1};
	omp_set_num_threads(4);
	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		pooh(ID, A);
	}
	printf("done!\n");
	return 0;
}
