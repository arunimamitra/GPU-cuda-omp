#include <iostream>
#include <omp.h>
#include <math.h>

void writeToFile(bool* compositeNumbers, int N, FILE* fw){
    for(int i = 2; i < N; i++){
        if(!compositeNumbers[i]){
            fprintf(fw,"%ld \n",i);
        }
    }
}

int main(int argc, char** argv) 
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
    #pragma omp target data map (tofrom: compositeNumbers[0:N]) 
    for (int i=2;i<=sqrt(N); i++){
      if (!compositeNumbers[i]){    /* Number is a prime number - assign multiples of the number true */
        #pragma omp target teams distribute parallel for
        /* Parallely allocating true value to all multiples prime numbbers */
        for (int j=i*i; j<= N; j += i){
          compositeNumbers[j]=true;
        }
      }
    }
    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(compositeNumbers, N, fw);
}