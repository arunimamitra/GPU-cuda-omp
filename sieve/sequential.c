#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

void writeToFile(bool *compositeNumbers, int N, FILE *fw)
{
    for (int i = 2; i <= N; i++)
    {
        if (compositeNumbers[i] == false)
            fprintf(fw,"%ld \t",i);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("usage: Executable name\n");
        printf("Parameter N\n");
        printf("name = The name of the output file\n");
        exit(1);
    }

	long int N = atoi(argv[1]);
	bool *compositeNumbers = (bool *)calloc(N+1, sizeof(bool));

    for(long int i=2; i <= N/2; i++)
    {
        if(!compositeNumbers[i]) /* Number is a prime number - assign multiples of the number true */
        {
            int j=2;
            while(i*j<=N)
            {
                compositeNumbers[i*j]=true;
                j++;
            }
        }
    }

    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(compositeNumbers, N, fw);
    return 0;
}