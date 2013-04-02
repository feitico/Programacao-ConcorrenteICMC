#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>

#define A 0
#define B 1
#define T 2
#define P 3

#define BITS_PER_DIGIT 3.32192809488736234787

int main() {
	mpf_t params[4][2]; /* Parameters: 0 - a, 1 - b, 2 - t, 3 - p*/
	mpf_t pi[2];
	mpf_t real_pi; /* Used to test the 10 millions digits of pi */
	mpf_t error;
	int i, j;
	clock_t start, end;
	double cpu_time_used;
	char filename[20]; /* Name of the file gauss_seq_it-%.txt*/
	FILE *file; /* pointer to the output file */
	FILE *filePi; /* File with real pi digits */
	FILE *fileTime; /* File used to write the execution time */

	/* Set default precision - 11 million */
	mpf_set_default_prec(BITS_PER_DIGIT * 11000000);
	
	/* Initialization */
	for(i=0; i<4; i++)
		for(j=0; j<2; j++)
			mpf_init(params[i][j]);
	mpf_init_set_d(pi[0], 0.0);
	mpf_init(pi[1]);
	mpf_init(real_pi);
	mpf_init_set_str(error, "1e-10000000", 10);

	/* Initial value setting */
	mpf_set_d(params[A][0], 1.0); /* a0 = 1*/
	
	mpf_sqrt_ui(params[B][0], 2); /* b0 = sqrt(2) */
	mpf_ui_div(params[B][0], 1, params[B][0]); /* b0 = 1 / sqrt(2) */
	
	mpf_set_d(params[T][0], 0.25); /* t0 = 1 / 4 */

	mpf_set_d(params[P][0], 1.0); /* p0 = 1 */


	/* Load the reals digits of pi */
	filePi = fopen("pi.txt", "r");
	gmp_fscanf(filePi, "%Ff", real_pi); 
	fclose(filePi);

	i=1;
	j=0;
	
	start = clock();	

	/* Iterations */	
	do {
/*	for(i = 1; i<25; i++) {*/
		j = (j + 1) % 2;
	
		/* a(i+1) = (a(i) + b(i)) / 2 */
		mpf_add(params[A][j], params[A][(j+1) % 2], params[B][(j+1) % 2]); /* a(i+1) = a(i)  + b(i) */
		mpf_div_ui(params[A][j], params[A][j], 2); /* a(i+1) = (a(i) + b(i)) / 2 */

		/* b(i+1) = sqrt(a(i) * b(i)) */
		mpf_mul(params[B][j], params[A][(j+1) % 2], params[B][(j+1) % 2]); /* b(i+1) = a(i) * b(i) */
		mpf_sqrt(params[B][j], params[B][j]); /* b(i+1) = sqrt (a(i) * b(i) */

		/* t(i+1) = t(i) - p(i) * (a(i) - a(i+1)) ^ 2 */
		mpf_sub(params[T][j], params[A][(j+1) % 2], params[A][j]); /* t(i+1) = a(i) - a(i+1) */
		mpf_pow_ui(params[T][j], params[T][j], 2); /* t(i+1) = (a(i) - a(i+1))^2*/
		mpf_mul(params[T][j], params[T][j], params[P][(j+1) % 2]); /* t(i+1) = p(i) * (a(i) - a(i+1))^2 */
		mpf_sub(params[T][j], params[T][(j+1) % 2], params[T][j]); /* t(i+1) = t(i) - p(i) * (a(i) - a(i+1))^2 */

		/* p(i+1) = 2 * p(i) */
		mpf_mul_ui(params[P][j], params[P][(j+1) % 2], 2); /* p(i+1) = 2 * p(i) */
		
		/* pi = ((a(i) + b(i))^2) / (4*t(i)) */
		mpf_add(pi[j], params[A][j], params[B][j]); /* pi = a(i) + b(i) */
		mpf_pow_ui(pi[j], pi[j], 2); /* pi = (a(i) + b(i))^2 */
		mpf_div_ui(pi[j], pi[j], 4); /* pi = ((a(i)+b(i))^2) / 4 */
		mpf_div(pi[j], pi[j], params[T][j]); /* *pi = ((a(i) + b(i))^2) / (4*t(i)) */
		
		/* Print the pi value in this iteration */
		sprintf(filename, "it-%d.txt", i++);
		file = fopen(filename, "w");
		gmp_fprintf(file, "%.10000000Ff", pi[j]);
		fclose(file);

		/* Calculate the error */
		mpf_sub(pi[(j+1)%2], real_pi, pi[j]);
		mpf_abs(pi[(j+1) % 2], pi[(j+1) % 2]);

		/* Stop interaction if real_pi is equal calculated pi */
	} while(mpf_cmp(pi[(j+1) % 2], error) >= 0);

	end = clock();
    	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	fileTime = fopen("execution_time.txt", "w");
	fprintf(fileTime, "Execution time: %f\n", cpu_time_used);
	fclose(fileTime);

	/* Clean up*/
	for(i=0; i<4; i++)
		for(j=0; j<2; j++)
			mpf_clear(params[i][j]);
	mpf_clear(pi[0]);
	mpf_clear(pi[1]);
	mpf_clear(real_pi);
	mpf_clear(error);

	return 0;
}
