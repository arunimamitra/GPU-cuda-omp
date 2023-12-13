#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<time.h>

using namespace std;

#define INF 10000000                    // mat[u][v]=INF represents that there is no edge between u and v
#define index(i, j, N)  ((i)*(N)) + (j)  // convert 2d index to 1D index

int *mat;  // matrix array
int *dist; // Array to store distance
long int N;     // Number of edges
bool negative_cycle; //flag for negative cycles

int readInputFile(string filename)
{
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        fprintf(stderr,"Unable to open the file....Exiting!");
        exit(1);
    }
    inputFile >> N;
    mat = (int *) malloc(N * N * sizeof(int));
    if(!mat)
    {
        fprintf(stderr,"Unable to allocate graph matrix of size %d x %d\n",N,N);
        exit(1);
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inputFile >> mat[index(i, j, N)];
    return 0;
}

int printDistance(string filename)
{
    ofstream output(filename);
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

void bellman_ford()
{
    negative_cycle = false;
    for (int u = 0; u < N; u++) {
        dist[u]=INF;
    }
    dist[0]=0;
    for (int i = 0; i < N - 1; i++)
    {
        #pragma omp target map(to: mat[0:N*N],N) map(tofrom: dist[0:N], negative_cycle)
        #pragma omp teams distribute parallel for collapse(2)
        for (int u = 0; u < N; u++) {
            for (int v = 0; v < N; v++) {
                int edge_weight = mat[index(u, v, N)];
                if (edge_weight < INF) {
                    if (dist[v] > dist[u] + edge_weight) {
                        #pragma omp atomic write
                        dist[v] = dist[u] + edge_weight;
                    }
                }
            }
        }
        #pragma omp barrier
    }
    //Check for negative cycles
    #pragma omp target map(to: mat[0:N*N], N) map(tofrom: dist[0:N], negative_cycle)
    #pragma omp teams distribute parallel for collapse(2)
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            int edge_weight = mat[index(u, v, N)];
            if (edge_weight < INF) {
                if (dist[v] > dist[u] + edge_weight) {
                    #pragma omp atomic write
                    negative_cycle = true;
                }
            }
        }
    }

}

int main(int argc, char *argv[])
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