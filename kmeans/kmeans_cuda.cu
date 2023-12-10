#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<float.h>
#include<cuda.h>
#include<cstring>
using namespace std;
#define index(i, j, N)  ((i)*(N)) + (j)

long int N;
float *points;
float *centroids;
int *clusters;                  // array of size N where each point is assigned a cluster idx 0,1....K-1
int *cluster_size;              // array of size K where cluster_size[i]=No of points assigned to Cluster<i> --> used to calculate new centroids
int iterations;
int K;


int readInputFile(string filename)
{
    ifstream inputFile(filename.c_str());
    if (!inputFile.is_open())
    {
        fprintf(stderr,"Unable to open the file....Exiting!");
        exit(1);
    }
    inputFile >> N;         // read N which is number of points
    points = (float *) malloc(N * 2 * sizeof(float));
    if(!points)
    {
        fprintf(stderr,"Unable to allocate Points vector of size %d x 2\n",N);
        exit(1);
    }
    for (long int i = 0; i < N; i++)
        for (int j = 0; j < 2; j++)
            inputFile >> points[index(i, j, 2)];
    inputFile.close();
    return 0;
}

void generate_initial_centroids()                    // generate K centroids
{
    centroids=(float *) malloc(K * 2 * sizeof(float));
    if(!centroids)
    {
        fprintf(stderr,"Unable to allocate random centroid vector of size %d x 2\n",K);
        exit(1);
    }
    // For first iteration, the first K coordinates are taken as centroids to mantain uniformity for testing.
    for(int i=0;i<K;i++)
    {
        centroids[index(i,0,2)]=points[index(i,0,2)];
        centroids[index(i,1,2)]=points[index(i,1,2)];
    }
}

void printResults(string filename)
{
    ofstream output(filename.c_str());
    output << "Total iterations taken = " << iterations << "\n";
    output << "Number of points N = "<< N <<"\n";
    output << "Centroids are:\n";
    for(int i=0;i<K;i++)
    {
        output << centroids[index(i,0,2)] << " , " << centroids[index(i,1,2)] << "\n";
    }
    output << "Clusters indices of points are:\n";
    for(long int i=0;i<N;i++)
    {
        output << clusters[i] << " ";
    }
    output.flush();
    output.close();
    return;
}

__global__ void assign_points_to_cluster_kernel(long int N, int K,int stride,float *points_d,float *centroids_d, int *clusters_d,int *cluster_size_d, bool *change_flag_d)
{
    long int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid==0) // first time called
    {
        change_flag_d[0]=false;
        for(long int i=0;i<K;i++)
            cluster_size_d[i]=0;
    }
    __syncthreads();

    for(long int j=tid*stride;j<(tid+1)*stride && j<N;j++)  //iterate over points
    {
        float min_dist=FLT_MAX;
        long int idx_closest_centroid=0;
        for(int c=0;c<K;c++)    //iterate over each centroid
        {
            float x=(centroids_d[index(c,0,2)]-points_d[index(j,0,2)])*(centroids_d[index(c,0,2)]-points_d[index(j,0,2)]);
            float y=(centroids_d[index(c,1,2)]-points_d[index(j,1,2)])*(centroids_d[index(c,1,2)]-points_d[index(j,1,2)]);
            if((x+y)<min_dist)
            {
                min_dist=(x+y);
                idx_closest_centroid=c;
            }
        }
        if(clusters_d[j]!=idx_closest_centroid)       // cluster of point i has changed
        {
            clusters_d[j]=idx_closest_centroid;
            change_flag_d[0]=true;
        }
        atomicAdd(&cluster_size_d[idx_closest_centroid],1); // use atomic add because simultaneous updates are happening
        //cluster_size[idx_closest_centroid]++;
    }
}

__global__ void calculate_new_centroids_kernel(long int N,int K,int stride,float *points_d,float *centroids_d,int *clusters_d,int *cluster_size_d)
{
    long int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid==0) // first thread launch --> reset all centroids
    {
        for(int i=0;i<K;i++)
        {
            centroids_d[index(i,0,2)]=0.0;
            centroids_d[index(i,1,2)]=0.0;
        }
    }

    __syncthreads(); // wait for the above code block to be completed, all other threads sit idle.
    for(long int j=tid*stride;j<(tid+1)*stride && j<N;j++)  //iterate over points
    {
        int cluster_current_point=clusters_d[j];
        atomicAdd(&centroids_d[index(cluster_current_point,0,2)],points_d[index(j,0,2)]);
        atomicAdd(&centroids_d[index(cluster_current_point,1,2)],points_d[index(j,1,2)]);
    }

    __syncthreads(); // wait for threads to complete sum before dividing

    if(tid==0)      // division is being done by just one thread
    {
        for(int i=0;i<K;i++)
        {
            if(cluster_size_d[i]>0)
            {
                centroids_d[index(i,0,2)]/=cluster_size_d[i];
                centroids_d[index(i,1,2)]/=cluster_size_d[i];
            }
        }
    }


}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: kmeans_seq input_file output_file K(optional)\n");
        fprintf(stderr, "input_file= path to input_file with graph\n");
        fprintf(stderr, "output_file=filename to store distance from source vertex\n");
        fprintf(stderr, "K=number of clusters, default set to 3\n");
        exit(1);
    }
    string input_filename=argv[1];
    string output_filename=argv[2];
    if(argc==3)                         // default K=3
        K=3;
    else
        K=stoi(argv[3]);
    if(readInputFile(input_filename))
        exit(1);
    generate_initial_centroids();
    clusters=(int *)malloc(sizeof(int)*N);
    if(!clusters)
    {
        fprintf(stderr,"Unable to allocate clusters vector of size %d\n",N);
        exit(1);
    }
    for(int i=0;i<N;i++)
        clusters[i]=0;
    cluster_size=(int *)malloc(sizeof(int)*K);
    if(!cluster_size)
    {
        fprintf(stderr,"Unable to allocate cluster size vector of size %d\n",K);
        exit(1);
    }
    bool *change_flag=(bool*)malloc(sizeof(bool));
    change_flag[0]=true;
    iterations =0;

    float *points_d,*centroids_d;
    int *clusters_d, *cluster_size_d;
    bool *change_flag_d;

    cudaError_t err = cudaSuccess;
    err=cudaMalloc((void **)&points_d,N*2*sizeof(float));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate point vector on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void**)&centroids_d,K*2*sizeof(float));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate centroid vector on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void**)&clusters_d,N*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate cluster vector on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void**)&cluster_size_d,K*sizeof(int));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate cluster_size vector on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err=cudaMalloc((void**)&change_flag_d,sizeof(bool));
    if(err != cudaSuccess)
    {
      fprintf(stderr, "Unable to allocate change flag boolean on device: %s\n", cudaGetErrorString(err));
      exit(1);
    }

    dim3 grid_size(8, 1, 1);
    dim3 block_size(512, 1, 1);
    int stride = ceil(N / (float)(grid_size.x * block_size.x));     // each thread will take care of 'stride' coordinates.. If stride=2, each thread computes for two coordinates
    cudaMemcpy(points_d, points, N*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_d, centroids, K *2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(clusters_d, clusters, N* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_size_d, cluster_size_d, K* sizeof(int), cudaMemcpyHostToDevice);

    while(change_flag[0])
    {
        assign_points_to_cluster_kernel<<<grid_size,block_size>>>(N,K,stride,points_d,centroids_d,clusters_d,cluster_size_d,change_flag_d);
        cudaDeviceSynchronize(); // wait for first kernel to complete
calculate_new_centroids_kernel<<<grid_size,block_size>>>(N,K,stride,points_d,centroids_d,clusters_d,cluster_size_d);
        cudaDeviceSynchronize(); // wait for second kernel to complete
        iterations++;
        cudaMemcpy(change_flag, change_flag_d, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(clusters,clusters_d, N* sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids,centroids_d, K*2* sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(points_d);
    cudaFree(centroids_d);
    cudaFree(clusters_d);
    cudaFree(cluster_size_d);
    cudaFree(change_flag_d);    
    printResults(output_filename);
    free(points);
    free(centroids);
    free(cluster_size);
    free(clusters);
    return 0;
}