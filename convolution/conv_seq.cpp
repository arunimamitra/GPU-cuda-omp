#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<float.h>

using namespace std;
#define index(i, j, N)  ((i)*(N)) + (j)


int *matrix;
int N;
int *filter;
int *output;


//initialize a unit matrix
void initialize_matrix()
{
    matrix = (int *) malloc(N * N * sizeof(int));
    if(!matrix)
    {
        fprintf(stderr,"Unable to allocate matrix of size %d x %d\n",N,N);
        exit(1);
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[index(i, j, N)]=1;
}
void initialize_filter()
{
    filter=(int *)malloc(3*3*sizeof(int));
    if(!filter)
    {
        fprintf(stderr,"Unable to allocate convolution filter of size 3 x 3\n");
        exit(1);
    }
    filter[index(0,0,3)]=1;
    filter[index(0,1,3)]=0;
    filter[index(0,2,3)]=-1;
    filter[index(1,0,3)]=2;
    filter[index(1,1,3)]=0;
    filter[index(1,2,3)]=-2;
    filter[index(2,0,3)]=1;
    filter[index(2,1,3)]=0;
    filter[index(2,2,3)]=-1;
}

void convolution()
{
    for (int i = 0;i<N-2;i++)
    {
        for (int j=0;j<N-2;j++)
        {
            int sum = 0;
            for (int m=0;m<3;m++)
            {
                for (int n=0;n<3;n++)
                {
                    sum += matrix[index(i + m, j + n, N)] * filter[index(m, n, 3)];
                }
            }
            output[index(i,j,N-2)] = sum;
        }
    }
}
void printResults(string filename)
{
    ofstream outputf(filename.c_str());
    for(int i=0;i<N-2;i++)
    {
        for(int j=0;j<N-2;j++)
        {
            outputf << output[index(i,j,N-2)] << " ";
        }
        outputf << "\n";
    }
    outputf.flush();
    outputf.close();
    return;
}

int main(int argc,char**argv) 
{
    if (argc < 3) 
    {
        fprintf(stderr, "usage: conv_seq N output_file \n");
        fprintf(stderr, "N=  size of input matrix\n");
        fprintf(stderr, "output_file=filename to store convoluted matrix\n");
        exit(1);
    }
    N=stoi(argv[1]);
    string output_filename=argv[2];
    initialize_matrix();
    initialize_filter();
    output=(int *)malloc((N-2)*(N-2)*sizeof(int));
    if(!output)
    {
        fprintf(stderr,"Unable to allocate output matrix of size %d x %d\n",N-2,N-2);
        exit(1);
    }
    double time_taken;
    clock_t start, end;
    start=clock();
    convolution();
    end=clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
    printResults(output_filename);
    free(matrix);
    free(filter);
    free(output);
    return 0;
}
