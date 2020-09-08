#include <stdio.h>
#include <stdlib.h>

int main()
{
    int i=0;
    float h[40], x[40], y=0;

    for (i=0; i<40; i++)
    {
        x[i]=2*i-1;
        h[i] = 40 - i;
        y = y + x[i]*h[i];
    }

        printf("Le resultat est :\t%lf",y);
        printf("\n");

}
