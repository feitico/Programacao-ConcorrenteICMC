#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>

#define BITS_PER_DIGIT 3.32192809488736234787

int main() {
	mpf_t x, y, aux, pi, error, epsilon, real_pi;
	int in, total;
	int i;
	clock_t start, end;
	double cpu_time_used;
	char filename[20]; /* Name of the file gauss_seq_it-%.txt*/
	FILE *file; /* pointer to the output file */
	FILE *filePi; /* File with real pi digits */
	gmp_randstate_t randstate;

	/* Set default precision - 11 million */
	mpf_set_default_prec(BITS_PER_DIGIT * 11000000);
	
	/* Init the rand state variable */
	gmp_randinit_default(randstate);

	/* Initialization */
	mpf_init(x);
	mpf_init(y);
	mpf_init(aux);
	mpf_init(pi);
	mpf_init(error);
	mpf_init_set_str(epsilon, "1e-10000000", 10);
	mpf_init(real_pi);
	in = total = 0;

	/* Load the reals digits of pi */
	filePi = fopen("pi.txt", "r");
	gmp_fscanf(filePi, "%Ff", real_pi); 
	fclose(filePi);


	/* Iterations */	
	for(i=1; i<10000; i++) {
		if(i % 200 == 1)
			start = clock();

		/* Generate a random point inside the square */
		mpf_urandomb(x, randstate, BITS_PER_DIGIT * 10000000);
		mpf_mul_ui(x, x, 2);
		mpf_sub_ui(x, x, 1); 
		mpf_urandomb(y, randstate, BITS_PER_DIGIT * 10000000);
		mpf_mul_ui(y, y, 2);
		mpf_sub_ui(y, y, 1);

		mpf_mul(x, x, x); /* x = x*x */
		mpf_mul(y, y, y); /* y = y*y */
		mpf_add(x, x, y); /* x = x*x + y*y */

		if(mpf_cmp_ui(x, 1) <= 0)  /* Check if the point is inside the circle of radius 1 */
			in++;
		total++;

		/* Calculate pi only each a hundred iterations */
		if(i % 200 == 0) {
			mpf_set_ui(pi, in);
			mpf_div_ui(pi, pi, total);
			mpf_mul_ui(pi, pi, 4);

			/* Print the pi value in this iteration */
			sprintf(filename, "it-%d.txt", i / 100);
			file = fopen(filename, "w");
			gmp_fprintf(file, "%.10000000Ff", pi);
			fclose(file);
			
			/* Calculate the error */
			mpf_sub(error, real_pi, pi);
			mpf_abs(error, error);

			/* If the error is lower than epsilon, however this won't happen so fast*/
			if(mpf_cmp(error, epsilon) < 0)
				break;

			end = clock();
    		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			printf("200 points - execution time %f\n", cpu_time_used);
		}
	}

	/* Clean up*/
	mpf_clear(x);
	mpf_clear(y);
	mpf_clear(pi);
	mpf_clear(error);

	return 0;
}
