#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <gmp.h>

#define A 0
#define Y 1

#define BITS_PER_DIGIT 3.32192809488736234787


pthread_mutex_t mutexA;

/* Borwein variables */
mpf_t params[2][2]; /* Parameters: 0 - a, 1 - y */
mpf_t y0Aux, y0Aux2, a0Aux, a0Aux2;
mpf_t pi[2]; /* PI value */
mpf_t real_pi;
mpf_t error;
int j; /* Used to alternate between new and old variables */
int iteracoes, x;

void* calc_a(void* args) {
	/*params[Y][1] * ( 1 + params[Y][1] + params[Y][1] ^ 2 )*/
	pthread_mutex_lock(&mutexA);	
	
	mpf_pow_ui(a0Aux, params[Y][1], 2);
        mpf_add(a0Aux, a0Aux, params[Y][1]);
        mpf_add_ui(a0Aux, a0Aux, 1);
        mpf_mul(a0Aux, params[Y][1], a0Aux);
	
	pthread_mutex_unlock(&mutexA);
        pthread_exit(NULL);
}

void* calc_b(void* args) {
	/*2 ^ ( 2 * i + 3 )*/
	mpf_set_ui(a0Aux2, 2);
        x = (2*iteracoes) + 3;
        mpf_pow_ui(a0Aux2, a0Aux2, x);

	pthread_exit(NULL);
}

int main() {
	pthread_t thread_a, thread_b; /* My threads*/
	int i;
	FILE *filePi, *fileTime;
	clock_t start, end;
	double cpu_time_used;
	mpf_set_default_prec(BITS_PER_DIGIT * 11000000);	
	
	/* Borwein Variable Initialization */   	
	for(i=0; i<2; i++)
    		for(j=0; j<2; j++)
    			mpf_init(params[i][j]);
    	
	mpf_init(real_pi);
	mpf_init(y0Aux);
	mpf_init(y0Aux2);
	mpf_init(a0Aux);
	mpf_init(a0Aux2);
	mpf_init(pi[0]);
    	mpf_init(pi[1]);	
	
	mpf_init_set_str(error, "1e-10000000", 10);

	/* Initial value setting */
	mpf_sqrt_ui(params[A][0], 2.0); /* a0 = sqrt(2)*/
	mpf_mul_ui(params[A][0], params[A][0], 4.0); /* a0 = 4 * sqrt(2) */
	mpf_ui_sub(params[A][0], 6.0, params[A][0]); /* a0 = 6 - 4 * sqrt(2) */ 
	
	mpf_sqrt_ui(params[Y][0], 2.0); /* y0 = sqrt(2) */
	mpf_sub_ui(params[Y][0], params[Y][0], 1.0); /* y0 = sqrt(2) - 1 */
	
	mpf_set_ui(pi[0], 0);
	mpf_set_ui(pi[1], 0);
	i = 1; 
	j = 1;
	iteracoes = 0;
	x = 0;

	/* Load the reals digits of pi */
	filePi = fopen("pi.txt", "r");
	gmp_fscanf(filePi, "%Ff", real_pi); 
	fclose(filePi);
	
	start = clock();

	while(1){
		/* y = ( 1 - (1 - y0 ^ 4) ^ 0.25 ) / ( 1 + ( 1 - y0 ^ 4) ^ 0.25 ) */
		mpf_pow_ui(y0Aux, params[Y][0], 4); 
		mpf_ui_sub(y0Aux, 1.0, y0Aux);
		mpf_sqrt(y0Aux, y0Aux);
                mpf_sqrt(y0Aux, y0Aux);

		mpf_add_ui(y0Aux2, y0Aux, 1.0);
		mpf_ui_sub(y0Aux, 1.0, y0Aux);
		
		mpf_div(params[Y][1], y0Aux, y0Aux2);

		/* a = a0 * ( 1 + params[Y][1] ) ^ 4 - 2 ^ ( 2 * i + 3 ) * params[Y][1] * ( 1 + params[Y][1] + params[Y][1] ^ 2 ) */
		/* Threads creation */                
		pthread_create(&thread_a, NULL, calc_a, NULL);
        	pthread_create(&thread_b, NULL, calc_b, NULL);  		

		pthread_join(thread_a, NULL);
                pthread_join(thread_b, NULL);
		
		/* 2 ^ ( 2 * i + 3 ) * params[Y][1] * ( 1 + params[Y][1] + params[Y][1] ^ 2 ) */	
		mpf_mul(a0Aux, a0Aux, a0Aux2);

		/*a0 * ( 1 + params[Y][1] ) ^ 4*/
                mpf_add_ui(a0Aux2, params[Y][1], 1);
                mpf_pow_ui(a0Aux2, a0Aux2, 4);
                mpf_mul(a0Aux2, params[A][0], a0Aux2);
	
		/* form the entire expression */
                mpf_sub(params[A][1], a0Aux2, a0Aux);
	
                mpf_set(params[A][0], params[A][1]);
                mpf_set(params[Y][0], params[Y][1]);

                mpf_ui_div(pi[j], 1, params[A][0]);
		gmp_printf("\nIteracao %d  | pi = %.25Ff", iteracoes, pi[j]);
		
		/* Calculate the error */
		mpf_sub(pi[(j+1)%2], real_pi, pi[j]);
		mpf_abs(pi[(j+1) % 2], pi[(j+1) % 2]);

		if(mpf_cmp(pi[(j+1)%2], error) < 0){
			printf("\n%d iteracoes para alcancar 10 milhoes de digitos de corretos.", iteracoes);			
			break;
		}

		j = (j+1) % 2;
		
		iteracoes++;
		i++;
	}

	end = clock();
    	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	fileTime = fopen("execution_time.txt", "w");
	fprintf(fileTime, "Execution time: %f\n", cpu_time_used);
	fclose(fileTime);
	
	/* Clean up*/
    	for(i=0; i<2; i++)
    		for(j=0; j<2; j++)
    			mpf_clear(params[i][j]);
    	
	mpf_clear(pi[0]);
    	mpf_clear(pi[1]);
	mpf_clear(real_pi);
	mpf_clear(error);

	return 0;
}
