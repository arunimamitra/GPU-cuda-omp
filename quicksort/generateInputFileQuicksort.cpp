#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include<iostream>


int main(int argc, char* argv[]) {

    FILE *fp;

    long long m = std::atoi(argv[1]);         // number of elements in file
    fp = fopen(argv[2], "a");
    //std::cout<<argv[2];
    //std::cout<<"\n"<<argv[1];    
long long i = 0;
    
    while(i<m){
	i++;
	int x = rand() % m;
	fprintf(fp, "%d\n", x);
		}
	    fclose(fp);
}
