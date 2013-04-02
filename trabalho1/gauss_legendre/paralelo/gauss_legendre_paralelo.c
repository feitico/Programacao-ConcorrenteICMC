#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <gmp.h>

#define A 0
#define B 1
#define T 2
#define P 3

#define BITS_PER_DIGIT 3.32192809488736234787

/* My semaphores*/
sem_t mutex_a0, mutex_a, mutex_b, mutex_b0, mutex_t, mutex_t0, mutex_p, mutex_p0;

/* Gauss Legendre variables */
mpf_t params[4][2]; /* Parameters: 0 - a, 1 - b, 2 - t, 3 - p*/
mpf_t pi[2]; /* PI value */
mpf_t real_pi;
mpf_t error;
int j; /* Used to alternate between new and old variables */

void* calc_a(void* args) {
	int i=0;
	while(1) {
		sem_wait(&mutex_a0);
		/* 
			Calculate variable A
			a(i+1) = (a(i) + b(i)) / 2 
		*/
		mpf_add(params[A][j], params[A][(j+1) % 2], params[B][(j+1) % 2]); /* a(i+1) = a(i)  + b(i) */
		mpf_div_ui(params[A][j], params[A][j], 2); /* a(i+1) = (a(i) + b(i)) / 2 */

		sem_post(&mutex_a);
		sem_post(&mutex_a);
	}
	return NULL;
}

void* calc_b(void* args) {
	int i=0;
	while(1) {
		sem_wait(&mutex_b0);		
		/* 
			Calculate variable B
			b(i+1) = sqrt(a(i) * b(i)) 
		*/
		mpf_mul(params[B][j], params[A][(j+1) % 2], params[B][(j+1) % 2]); /* b(i+1) = a(i) * b(i) */
		mpf_sqrt(params[B][j], params[B][j]); /* b(i+1) = sqrt (a(i) * b(i) */

		sem_post(&mutex_b);
	}
	return NULL;
}

void* calc_t(void* args) {
	int i=0;
	while(1) {
		sem_wait(&mutex_t0);
		sem_wait(&mutex_a);
		/* 
			Calculate variable T
			t(i+1) = t(i) - p(i) * (a(i) - a(i+1)) ^ 2 
		*/
		mpf_sub(params[T][j], params[A][(j+1) % 2], params[A][j]); /* t(i+1) = a(i) - a(i+1) */
		mpf_pow_ui(params[T][j], params[T][j], 2); /* t(i+1) = (a(i) - a(i+1))^2*/
		mpf_mul(params[T][j], params[T][j], params[P][(j+1) % 2]); /* t(i+1) = p(i) * (a(i) - a(i+1))^2 */
		mpf_sub(params[T][j], params[T][(j+1) % 2], params[T][j]); /* t(i+1) = t(i) - p(i) * (a(i) - a(i+1))^2 */
		
		sem_post(&mutex_t);
	}
	return NULL;
}

void* calc_p(void* args) {
	int i=0;
	while(1) {
		sem_wait(&mutex_p0);
		/* 
			Calculate variable P
			p(i+1) = 2 * p(i) 
		*/
		mpf_mul_ui(params[P][j], params[P][(j+1) % 2], 2); /* p(i+1) = 2 * p(i) */
		
		sem_post(&mutex_p);
	}
	return NULL;
}

int main() {
	pthread_t thread_a, thread_b, thread_t, thread_p; /* My threads*/
	int i;
	char filename[20];
	FILE *file, *filePi, *fileTime;
	clock_t start, end;
	double cpu_time_used;

	mpf_set_default_prec(BITS_PER_DIGIT * 11000000);	

	/* Gauss Legendre Variable Initialization */
    for(i=0; i<4; i++)
    	for(j=0; j<2; j++)
    		mpf_init(params[i][j]);
    mpf_init_set_d(pi[0], 0.0);
    mpf_init_set_d(pi[1], 0.0);
	mpf_init(real_pi);
	mpf_init_set_str(error, "1e-10000000", 10);
                                                                         
    /* Initial value setting */
    mpf_set_d(params[A][0], 1.0); /* a0 = 1*/
    mpf_sqrt_ui(params[B][0], 2); /* b0 = sqrt(2) */
    mpf_ui_div(params[B][0], 1, params[B][0]); /* b0 = 1 / sqrt(2) */	
    mpf_set_d(params[T][0], 0.25); /* t0 = 1 / 4 */
    mpf_set_d(params[P][0], 1.0); /* p0 = 1 */
	i = 1;
	j = 1;

	/* Semaphore initialization */
	sem_init(&mutex_a, 0, 0);
	sem_init(&mutex_a0, 0, 1);
	sem_init(&mutex_b, 0, 0);
	sem_init(&mutex_b0, 0, 1);
	sem_init(&mutex_t, 0, 0);
	sem_init(&mutex_t0, 0, 1);
	sem_init(&mutex_p, 0, 0);
	sem_init(&mutex_p0, 0, 1);

	/* Load the reals digits of pi */
	filePi = fopen("pi.txt", "r");
	gmp_fscanf(filePi, "%Ff", real_pi); 
	fclose(filePi);

	start = clock();

	/* Thread creation */
	pthread_create(&thread_a, NULL, calc_a, NULL);
	pthread_create(&thread_b, NULL, calc_b, NULL);
	pthread_create(&thread_t, NULL, calc_t, NULL);
	pthread_create(&thread_p, NULL, calc_p, NULL);

	while(1) {
		sem_wait(&mutex_a);
		sem_wait(&mutex_b);
		sem_wait(&mutex_p);
		
		/* pi = ((a(i) + b(i))^2) / (4*t(i)) */
		mpf_add(pi[j], params[A][j], params[B][j]); /* pi = a(i) + b(i) */
		mpf_pow_ui(pi[j], pi[j], 2); /* pi = (a(i) + b(i))^2 */
		mpf_div_ui(pi[j], pi[j], 4); /* pi = ((a(i)+b(i))^2) / 4 */

		sem_wait(&mutex_t); /* Wait variable t */
		mpf_div(pi[j], pi[j], params[T][j]); /* *pi = ((a(i) + b(i))^2) / (4*t(i)) */

		sprintf(filename, "it-%d.txt", i);
		file = fopen(filename, "w");
		gmp_fprintf(file, "%.10000000Ff", pi[j]);
		fclose(file);
		
		/* Calculate the error */
		mpf_sub(pi[(j+1)%2], real_pi, pi[j]);
		mpf_abs(pi[(j+1) % 2], pi[(j+1) % 2]);

		if(mpf_cmp(pi[(j+1)%2], error) < 0)
			break;

		j = (j+1) % 2;
		
		sem_post(&mutex_a0);
		sem_post(&mutex_b0);
		sem_post(&mutex_t0);
		sem_post(&mutex_p0);
	} 
	
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
