#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#include <omp.h>

int board_length;

bool isValid(int board[board_length]){
    for (int i = 0; i < board_length; i++){
        for(int j=i+1;j<board_length; j++){
            if (board[i]==board[j]) return false;
            else if (board[i]-board[j]==i-j || board[i]-board[j]==j-i) return false;
        }
    }
    return true;
}

int main (int argc, char **argv){
    if (argc != 2) {
        printf("usage: Executable name\n");
        printf("board length = The parameter N\n");
        exit(1);
    }
    board_length=atoi(argv[1]);
	int solutionsPossible = 0;
    
    omp_set_num_threads(32);
    long long int total_board_size=pow(board_length, board_length);
    #pragma omp parallel for
    for (int i = 0; i < total_board_size; i++){
        int column=i;
        int board[board_length];
        for (int j = 0; j < board_length; j++){
            board[j] = column % board_length;
            column/=board_length;
        }
    if (isValid(board)){
        #pragma omp atomic
        solutionsPossible++;
    }
    }
    printf("Board length = %d, Number of unique solutions to place all the queens = %d", board_length, solutionsPossible);
}
