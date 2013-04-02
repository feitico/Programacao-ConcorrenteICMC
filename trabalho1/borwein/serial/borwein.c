#include <stdio.h>
#include <gmp.h>
#include <time.h>

#define BITS_PER_DIGIT 3.32192809488736234787

int main ()
{
    
    int i = 0; /*Contador de iterações*/
    int k = 8; /*Fator de multiplicação*/
    mpf_t pi_pas, pi_novo;
    mpf_t a_pas, a_novo;
    mpf_t y_pas, y_novo;
    mpf_t temp1, temp2;
    mpf_t e;
    FILE *fileTime; /*Ponteiro do arquivo de saída*/
    clock_t start, end;
	double cpu_time_used;

    mpf_set_default_prec(BITS_PER_DIGIT * 11000000); /*Precisão default*/
    
    
    /*Inicialização das variáveis*/
    mpf_init(pi_pas);
    mpf_init(pi_novo);
    mpf_init(a_pas);
    mpf_init(y_pas);
    mpf_init(temp1);
    mpf_init(temp2);    
    mpf_init_set_d(a_novo, 32.0);
    mpf_sqrt(a_novo, a_novo);
    mpf_ui_sub(a_novo, 6, a_novo);
    mpf_init_set_d(y_novo, 2.0);
    mpf_sqrt(y_novo, y_novo);
    mpf_sub_ui(y_novo, y_novo, 1);    
    mpf_init_set_str(e, "1e-10000000", 10);    
    mpf_ui_div(pi_novo, 1, a_novo);
    
	start = clock();

    /*Calcula as iterações*/
    do
    {
        mpf_swap(pi_pas, pi_novo);
        mpf_swap(a_pas, a_novo);
        mpf_swap(y_pas, y_novo);
       
        mpf_pow_ui(y_pas, y_pas, 4);
        mpf_ui_sub(y_pas, 1, y_pas);
        mpf_sqrt(y_pas, y_pas);
        mpf_sqrt(y_pas, y_pas);
        mpf_add_ui(temp1, y_pas, 1);
        mpf_ui_sub(y_novo, 1, y_pas);
        mpf_div(y_novo, y_novo, temp1);
        
        mpf_add_ui(temp1, y_novo, 1);
        
        mpf_pow_ui(temp2, y_novo, 2);
        mpf_add(temp2, temp2, temp1);
        mpf_mul(temp2, temp2, y_novo);
        mpf_mul_ui(temp2, temp2, k);
        k *= 4;
        
        mpf_pow_ui(temp1, temp1, 4);
        mpf_mul(temp1, temp1, a_pas);
        mpf_sub(a_novo, temp1, temp2);
        
        mpf_ui_div(pi_novo, 1, a_novo);
        
        mpf_sub(pi_pas, pi_novo, pi_pas);
        mpf_abs(pi_pas, pi_pas);
        
	gmp_printf("\nIteracao %d  | pi = %.25Ff", i, pi_novo);
	i++;
    } while ( mpf_cmp(e, pi_pas) < 0 );
 	
	end = clock();
 	
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	fileTime = fopen("execution_time.txt", "w");
	fprintf(fileTime, "Execution time: %f\n", cpu_time_used);
	fclose(fileTime);	
  
    /*Libera espaço de memória*/
    mpf_clear(pi_pas);
    mpf_clear(pi_novo);
    mpf_clear(a_pas);
    mpf_clear(a_novo);
    mpf_clear(y_pas);
    mpf_clear(y_novo);
    mpf_clear(temp1);
    mpf_clear(temp2);
    mpf_clear(e);
    
    return 0;
}
