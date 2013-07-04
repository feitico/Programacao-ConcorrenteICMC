#include "jacobi.h" 

int main() {
    int j_order;
    int j_row_test;
    float j_error;
    int j_ite_max, qtdIteracoes;
    float** ma;
    float* mb;
    float* approx;
    float result = 0;

    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);
   
    ma = createMatrixFloat( j_order );
    mb = createVectorFloat( j_order );
    approx = createVectorFloat( j_order );

    /* reads the values of the matrix a */
    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%f", &ma[i][j]);
    
    /* assign the values of the first approximation */
    for( int i=0; i<j_order; i++ )
        approx[i] = 0;

    /* reads the values of the matrix b */
    for(int i=0; i<j_order; i++)
        scanf("%f", &mb[i]);
    
    qtdIteracoes = jacobiMethod( ma, approx, mb, j_order, j_ite_max, j_error );

    for( int i = 0; i < j_order; i++ )
        result += ma[j_row_test][i] * approx[i];
    
    printf("Iterations: %d\n", qtdIteracoes);
    printf("RowTest: %d => [%f] =? %f \n", j_row_test, result, mb[j_row_test]);

    deleteMatrix( ma, j_order );
    deleteVector( mb );
    deleteVector( approx );
    return 0;
}
