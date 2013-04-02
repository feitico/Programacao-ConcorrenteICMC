#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>

#define BITS_PER_DIGIT 3.32192809488736234787
#define NTHREADS 4

unsigned long int in[NTHREADS];
unsigned long int total[NTHREADS];

sem_t mutex_pi[NTHREADS], mutex_points[NTHREADS];

void* generate_points(void* threadid) {
	int id = *((int*) threadid);
	int i, j;
	mpf_t x, y;
	gmp_randstate_t randstate;

	mpf_init(x);
	mpf_init(y);
	gmp_randinit_default(randstate);
	gmp_randseed_ui(randstate, rand());

	for(i=0; i<100; i++) {
		sem_wait(&mutex_pi[id]);
		
		/* Each thread calculate 50 points in parallel */
		for(j=0; j<50; j++) {
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
            	in[id]++; /* Count the inner points */
            total[id]++;  /* Count the total points */
		}

		sem_post(&mutex_points[id]);
	}

	return NULL;
}

int main() {
	mpf_t pi, error, epsilon, real_pi;
	unsigned long int in_total, total_total; /* The sum of all the ins and totals*/
	int i, j;
	clock_t start, end;
	double cpu_time_used;
	char filename[20]; /* Name of the file gauss_seq_it-%.txt*/
	FILE *file; /* pointer to the output file */
	FILE *filePi; /* File with real pi digits */
	pthread_t threads[NTHREADS];

	/* Set default precision - 11 million */
	mpf_set_default_prec(BITS_PER_DIGIT * 11000000);

	/* Generate random numbers to become seed of the gnu mp random number generator*/
	srand(time(NULL));

	/* Initialization */
	mpf_init(pi);
	mpf_init(error);
	mpf_init_set_str(epsilon, "1e-10000000", 10);
	mpf_init(real_pi);
	
	/* Load the reals digits of pi */
	filePi = fopen("pi.txt", "r");
	gmp_fscanf(filePi, "%Ff", real_pi); 
	fclose(filePi);
	
	in_total = total_total = 0;
	for(i=0; i<NTHREADS; i++) {
		in[i] = total[i] = 0;
		sem_init(&mutex_pi[i], 0, 1);
		sem_init(&mutex_points[i], 0, 0);

	}
	
	
	for(i=0; i<NTHREADS; i++) {
		pthread_create(&threads[i], NULL, generate_points, (void*) &i);
	}

	/* Iterations */	
	for(i=0; i<100; i++) {
		start = clock();	
		
		/* Calculate pi only each a hundred iterations */
		for(j=0; j<NTHREADS; j++) 
			sem_wait(&mutex_points[j]); /* Wait thread j */

		/* Get inside points and total points informations of thread j */
		for(j=0; j<NTHREADS; j++) {
			in_total += in[j];
			in[j] = 0;
                                                                  
			total_total += total[j];
			total[j] = 0;;
		}
		
		mpf_set_ui(pi, in_total);
		mpf_div_ui(pi, pi, total_total);
		mpf_mul_ui(pi, pi, 4);

		/* Print the pi value in this iteration */
		sprintf(filename, "it-%d.txt", i);
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
			
		for(j=0; j<NTHREADS; j++) 
			sem_post(&mutex_pi[j]); /* Release thread j */
	}

	/* Clean up*/
	mpf_clear(pi);
	mpf_clear(error);
	mpf_clear(epsilon);

	return 0;
}
