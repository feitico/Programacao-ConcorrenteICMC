#include "jacobi.h" 

int main() {
    int j_order;
    int j_row_test;
    float j_error;
    int j_ite_max;
    float** ma;
    float* mb;
    float* approx;

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
    
    jacobiMethod( ma, approx, mb, j_order, j_ite_max, j_error );

    //printMatrix( ma, j_order );
    //printVector( mb, j_order );

    deleteMatrix( ma, j_order );
    deleteVector( mb );
    return 0;
}
