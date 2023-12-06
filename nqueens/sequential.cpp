#include <iostream>
#include <vector>
#include <string>
#include <cmath>

bool isValid(int* board, int j)
{
    for (int row = 0; row < j; ++row)
    {
        if (board[row] == board[j])    /* we cannot place queen in the same row twice */
            return false;
        int col1=board[j]-j+row;
        int col2=board[j]+j-row;
        if (board[row] == col1 || board[row] == col2)   /* checking left and right diagonals */
            return false;
    }
    return true;
}

int countPossibleSolutions(int board_length) {

    long long int total_board_size = powl(board_length, board_length);
    int solutionsPossible = 0;

     for (long long int i = 0; i < total_board_size; i++) {
        bool valid = true;
        int board[11];
        long long int column = i;   /* columns are always kept unique */

        for (int j = 0; j < board_length; j++) {
            board[j] = column % board_length;

            if (!isValid(board, j))   
                { 
                    valid=false;
                    break;
                }
            column/= board_length;}
            if (valid) { ++solutionsPossible; }
        }
    
    return solutionsPossible;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("usage: Executable name\n");
        printf("board length = The parameter N\n");
        exit(1);
    }
    int board_length=atoi(argv[1]);
	int solutionsPossible = countPossibleSolutions(board_length);
    printf("Board length = %d, Number of unique solutions to place all the queens = %d", board_length, solutionsPossible);
}