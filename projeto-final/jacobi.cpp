#include <stdio.h>
#include <stdlib.h>

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

    ma = (int**) malloc(j_order * sizeof(int*));
    for(int i=0; i<j_order; i++)
        ma[i] = (int*) malloc(j_order * sizeof(int));

    mb = (int*) malloc(j_order * sizeof(int));

    for(int i=0; i<j_order; i++)
        for(int j=0; j<j_order; j++)
            scanf("%d", &ma[i][j]);

    for(int i=0; i<j_order; i++)
        scanf("%d", &mb[i]);
    

    return 0;
}
