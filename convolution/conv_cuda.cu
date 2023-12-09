#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<float.h>
#include<cuda.h>


using namespace std;
#define index(i, j, N)  ((i)*(N)) + (j)


int *matrix;
int N;
int *filter;
int *output;
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
    // SOBEL FILTER
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

__global__ void convolution_kernel(int N,int *matrix_d,int *output_d,int *filter_d)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N-2 && j < (N-2))
    {
        int sum = 0;
        for (int m=0; m<3; ++m)
        {
            for (int n=0;n<3;++n)
            {
                sum += matrix_d[(i + m)*N+(j+ n)] * filter_d[m*3+n];
            }
        }
        output_d[i*(N-2)+j] = sum;
    }
    
}


int main(int argc,char**argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: conv_seq input_file output_file \n");
        fprintf(stderr, "N= size of input square matrix\n");
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
        fprintf(stderr,"Unable to allocate output matrix of size %d x %d\n",(N-2),(N-2));
        exit(1);
    }

    // Device data strcutures
    int *matrix_d;
    int *output_d;
    int *filter_d;
    cudaError_t err = cudaSuccess;
    err=cudaMalloc((void **)&matrix_d,N*N*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate matrix on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void **)&output_d,(N-2)*(N-2)*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate output matrix on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void **)&filter_d,3*3*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate filter matrix on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }

    dim3 grid_size(8, 1, 1);
    dim3 block_size(32, 32, 1);
    int w=N-2;
    cudaMemcpy(matrix_d,matrix,N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d,output,w*w*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d,filter,3*3*sizeof(int),cudaMemcpyHostToDevice);
    convolution_kernel<<<grid_size,block_size>>>(N,matrix_d,output_d,filter_d);
    cudaMemcpy(output,output_d, w*w*sizeof(int), cudaMemcpyDeviceToHost);
    printResults(output_filename);
    free(matrix);
    free(filter);
    free(output);
    cudaFree(matrix_d);
    cudaFree(output_d);
    cudaFree(filter_d);
    return 0;
}