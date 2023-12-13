#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 128

void writeToFile(bool* compositeNumbers, long int N, FILE* fw){
    for(long int i = 2; i < N; i++){
        if(!compositeNumbers[i])
            fprintf(fw,"%ld \n",i);
    }
}

__global__ void sieve(bool*, long int, long int, long int, long int);

int main(int argc, char** argv){
    
    if (argc != 3)
    {
        printf("usage: Executable name\n");
        printf("Parameter N\n");
        printf("name = The name of the output file\n");
        exit(1);
    }

    long int N = atoi(argv[1]);
	bool *compositeNumbers = (bool *)calloc(N+1, sizeof(bool));

    bool* device_compositeNumbers;
    cudaMalloc((void**)&device_compositeNumbers, (N+1)*sizeof(bool));
    cudaMemcpy(device_compositeNumbers, compositeNumbers, (N+1)*sizeof(bool), cudaMemcpyHostToDevice);  /* Copy H to D */

    for(long int i = 2; i <= sqrt(N); i++){
        if(compositeNumbers[i] == false){   /* Number is a prime number - assign multiples of the number true */
            long int start = i*i;
            long int end = N;
            long int step = i;
            long int num_blocks = (end-start)/step/BLOCK_SIZE + 1;
            sieve<<<num_blocks, BLOCK_SIZE>>>(device_compositeNumbers, start, end, step, N);
            cudaDeviceSynchronize();
        }
    }
    cudaMemcpy(compositeNumbers, device_compositeNumbers, (N+1)*sizeof(bool), cudaMemcpyDeviceToHost);  /* Copy D to H */
    cudaFree(device_compositeNumbers);
    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(compositeNumbers, N, fw);
}

/* each kernel takes one prime number  and allocates true value to all multiples of that prime numbber */
__global__ void sieve(bool* compositeNumbers, long int start, long int end, long int step, long int N){
    long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    long int multiple = start + idx*step;
    if(multiple <= N){
        compositeNumbers[multiple] = true;
    }
}