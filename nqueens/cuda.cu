#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define N_MAX 11

__global__ void kernelIsValid(int board_length, long long int total_board_size, long long int offset, int *d_solutions, int *d_num_solutions) {
    long long int column = (long long int)(threadIdx.x + blockIdx.x * blockDim.x) + offset;
    if (column>= total_board_size) return;      /* base condition */
    
    bool isValid;
    int board[N_MAX];

    for (int currRow = 0; currRow<board_length; currRow++) {
        board[currRow]=column%board_length;
        int j = board[currRow];
        isValid= true;
        for (int tmpRow = 0; tmpRow < currRow; tmpRow++)
        {
            if (board[tmpRow]==j) isValid=false;
            int col1 = j-currRow+tmpRow;
            int col2 = j+currRow-tmpRow;
            if (board[tmpRow] == col1 || board[tmpRow] == col2) isValid=false;
        }

        if (isValid==false) return;
        column/= board_length;
    }
    int index = atomicAdd(d_num_solutions,1);
}


int countPossibleSolutions(int board_length)
{
    int solutionsPossible = 0;
    int *deviceSolutions ;
    int *deviceSolutionsPossible;
    long long int total_board_size = powl(board_length,board_length);
    cudaMalloc(&deviceSolutionsPossible, sizeof(int));
    cudaMemcpy(deviceSolutionsPossible, &solutionsPossible, sizeof(int), cudaMemcpyHostToDevice);   /* Copy H to D */

    int id_offsets = 1; //initialise as 1 so that the kernel is executed at least once

    int grid = 256;
    int block = 64;
    if (total_board_size > grid * block)
    id_offsets = std::ceil((double) total_board_size / (grid * block));

    for (long long int i = 0; i < id_offsets; i++) {
        kernelIsValid<<<grid, block>>>(board_length, total_board_size, (long long int)grid * block * i, deviceSolutions, deviceSolutionsPossible); /* kernel call for 2d execution */
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&solutionsPossible, deviceSolutionsPossible, sizeof(int), cudaMemcpyDeviceToHost);   /* Copy D to H */
    cudaFree(deviceSolutionsPossible);
    return solutionsPossible;
}

int main(int argc, char** argv)
{
    if(argc != 2){
		std::cout<<"Command Line: ./program n"<<std::endl;
		std::cout<<"n is the size of the board"<<std::endl;
		exit(1);
	}
    int board_length=atoi(argv[1]);
	int solutionsPossible = countPossibleSolutions(board_length);
    printf("Board length = %d, Number of unique solutions to place all the queens = %d\n", board_length, solutionsPossible);
}