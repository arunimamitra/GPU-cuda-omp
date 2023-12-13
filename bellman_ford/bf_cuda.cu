#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<time.h>
#include<cuda.h>
#include<string.h>
#include<cstring>
using namespace std;

#define INF 10000000                    // mat[u][v]=INF represents that there is no edge between u and v
#define index(i, j, N)  ((i)*(N)) + (j)  // convert 2d index to 1D index

int *mat;  // matrix array
int *dist; // Array to store distance
long int N;     // Number of edges
bool negative_cycle; //flag for negative cycles
int num_blocks=8;
int num_threads=64;

int readInputFile(string filename)
{
    ifstream inputFile(filename.c_str());
    //inputFile.open(filename.c_str());
    if (!inputFile.is_open())
    {
        fprintf(stderr,"Unable to open the file....Exiting!");
        exit(1);
    }
    inputFile >> N;
    mat = (int *) malloc(N * N * sizeof(int));
    if(!mat)
    {
        fprintf(stderr,"Unable to allocate graph matrix of size %lx %l\n",N,N);
        exit(1);
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inputFile >> mat[index(i, j, N)];
    return 0;
}

int printDistance(string filename)
{
    ofstream output(filename.c_str());
    if(!negative_cycle)
    {
        for(int i=0;i<N;i++)
        {
            if(dist[i]>INF)
                dist[i]=INF;
            output << dist[i] << "\n";
        }
    }
    else
        output << "Graph has negative cycle\n";
    output.flush();
    output.close();
    return 0;
}


__global__ void bellman_ford_one_iter(int n, int *mat_d, int *dist_d, bool *distance_change_d,int iter)
{
        int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
        int elementSkip = blockDim.x * gridDim.x;

        if(global_tid >= n) return;
        for(int u = 0 ; u < n ; u ++){
                for(int v = global_tid; v < n; v+= elementSkip){
                        int weight = mat_d[u * n + v];
                        if(weight < INF){
                                int new_dist = dist_d[u] + weight;
                                if(new_dist < dist_d[v]){
                                        dist_d[v] = new_dist;
                                        *distance_change_d = true;
                                }
                        }
                }
        }

}

void bellman_ford()
{
    dim3 blocks(num_blocks);
    dim3 threads(num_blocks);

    int *mat_d, *dist_d,iter=0;
    bool distance_change, *distance_change_d;

    cudaError_t err = cudaSuccess;
    err=cudaMalloc(&mat_d,N*N*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate graph matrix on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc(&dist_d,N*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate distance array on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc(&distance_change_d,sizeof(bool));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate varibale distance_change on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }

    negative_cycle=false;
    for(int i=0;i<N;i++)
        dist[i]=INF;
    dist[0]=0;
    cudaMemcpy(mat_d, mat, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dist_d, dist, N*sizeof(int), cudaMemcpyHostToDevice);
    for(;;)
    {
        distance_change=false;
        cudaMemcpy(distance_change_d, &distance_change, sizeof(bool), cudaMemcpyHostToDevice);
        // Invoke kernel
        bellman_ford_one_iter<<<blocks,threads>>>(N,mat_d,dist_d,distance_change_d,iter);
        cudaDeviceSynchronize();
        cudaMemcpy(&distance_change, distance_change_d, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;
        if(iter>N-1)
        {
            negative_cycle=true;
            break;
        }
        if(!distance_change)
            break;
    }
    if (!negative_cycle)
        cudaMemcpy(dist, dist_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dist_d);
    cudaFree(mat_d);
    cudaFree(distance_change_d);
}


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: seq input_file output_file\n");
        fprintf(stderr, "input_file= path to input_file with graph\n");
        fprintf(stderr, "output_file=filename to store distance from source vertex\n");
        exit(1);
    }

    string input_filename=argv[1];
    string output_filemame=argv[2];
    if(readInputFile(input_filename))
        exit(1);
    dist=(int *)malloc(N* sizeof(int));
    if(!dist)
    {
        fprintf(stderr,"Unable to allocate distance matrix of size %d\n",N);
        exit(1);
    }
    bellman_ford();
    if(printDistance(output_filemame))
        exit(1);
    return 0;
}