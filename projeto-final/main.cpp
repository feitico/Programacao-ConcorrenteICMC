#include "jacobi.h" 

int main() {
    int j_order;
    int j_row_test;
    float j_error;
    int j_ite_max;
    int** ma;
    int* mb;

    scanf("%d", &j_order);
    scanf("%d", &j_row_test);
    scanf("%f", &j_error);
    scanf("%d", &j_ite_max);
   
    ma = createMatrix( j_order );
    mb = createVector( j_order );

    /* le os valores da matriz a */
    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%d", &ma[i][j]);
    
    /* le os valores da matriz b */
    for(int i=0; i<j_order; i++)
        scanf("%d", &mb[i]);
    
    printMatrix( ma, j_order );
    printVector( mb, j_order );

    deleteMatrix( ma, j_order );
    deleteVector( mb );
    return 0;
}
