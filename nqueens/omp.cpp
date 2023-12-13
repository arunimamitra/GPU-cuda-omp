#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#define MAX_N 11

int main(int argc, char* argv[])
{
	if(argc != 2){
		std::cout<<"Command Line: ./program n"<<std::endl;
		std::cout<<"n is the size of the board"<<std::endl;
		exit(1);
	}
    long long board_length = atoi(argv[1]);
    long long total_board_configurations_possible = pow(board_length,board_length);  
    int solutionsPossible = 0;
    
	long long queens[board_length]; 
	queens[0] = 1;
	for(int i = 1; i < board_length; i++){
		queens[i] = queens[i-1]*10;
	}

	#pragma omp target teams num_teams(512) map(tofrom:solutionsPossible) map(to:queens)
  	{
		#pragma omp distribute parallel for
		for (long long k = 0; k < total_board_configurations_possible; k++)
		{
			long long  ind2 = k;
			long long  board = 0;
			for (long long i = 0; i < board_length; i++)
			{
                board*= 10;
				board+= ind2 % board_length;
				ind2/= board_length;
			}
			bool isValid = true;
			for (long long i = 0; i < board_length; i++)    /* we cannot place queen in the same row twice */
			{
                long long queen1 = board / queens[i];
		        queen1 = queen1 % 10;
				for (long long j = i+1; j < board_length; j++)
				{
                    long long queen2 = board / queens[j];
		            queen2 = queen2 % 10;
					if (queen1 == queen2 || queen1 - queen2 == i - j || queen1 - queen2 == j - i){ /* checking column ; left and right diagonals */
						isValid = false;
						break;
					}
				}
				if(!isValid){
					break;
				}
			}
			if (isValid)
			{
				#pragma omp critical
				solutionsPossible++;
			}
		}
	}
	
    printf("Board length = %lld, Number of unique solutions to place all the queens = %d\n", board_length, solutionsPossible);

    
	return 0;
}
