#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<float.h>
#include<omp.h>

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

void generate_initial_centroids()                    // generate K random centroids
{
    centroids=(float *) malloc(K * 2 * sizeof(float));
    if(!centroids)
    {
        fprintf(stderr,"Unable to allocate random centroid vector of size %d x 2\n",K);
        exit(1);
    }
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
    output << "Clusters  indices of points are:\n";
    for(long int i=0;i<N;i++)
    {
        output << clusters[i] << " ";
    }
    output.flush();
    output.close();
    return;
}

bool assign_points_to_clusters()
{
    for(int i=0;i<K;i++)
        cluster_size[i]=0;
    bool cluster_change=false;      // keeps track of whether there was cluster change for any point
    // The following  outer loop has to be parallelized
    #pragma omp target teams num_teams(512) map(to: points[0:N*2],centroids[0:K*2]) map(tofrom: cluster_change, clusters[0:N],cluster_size[0:K])
    {
        #pragma omp distribute parallel for
            for(long int i=0;i<N;i++)    //iterate over points
            {
                float min_dist=FLT_MAX;
                int idx_closest_centroid=0;
                for(int j=0;j<K;j++)    // iterate over centroids
                {
                    float x=(centroids[index(j,0,2)]-points[index(i,0,2)])*(centroids[index(j,0,2)]-points[index(i,0,2)]);
                    float y=(centroids[index(j,1,2)]-points[index(i,1,2)])*(centroids[index(j,1,2)]-points[index(i,1,2)]);
                    if((x+y)<min_dist)
                    {
                        min_dist=(x+y);
                        idx_closest_centroid=j;
                    }
                }
                {
                if(clusters[i]!=idx_closest_centroid)       // cluster of point i has changed
                {
                    clusters[i]=idx_closest_centroid;
                    cluster_change=true;
                }
                #pragma omp critical
                {
                        cluster_size[idx_closest_centroid]++;

                }
                }
            }
    }
    return cluster_change;
}

void calculate_new_centroids()
{
    for(int i=0;i<K;i++)
    {
        centroids[index(i,0,2)]=0.0;
        centroids[index(i,1,2)]=0.0;
    }
    {
        for(long int i=0;i<N;i++)
        {
            int cluster_current_point=clusters[i];
            centroids[index(cluster_current_point,0,2)]= centroids[index(cluster_current_point,0,2)]+points[index(i,0,2)];
            centroids[index(cluster_current_point,1,2)]=centroids[index(cluster_current_point,1,2)]+points[index(i,1,2)];
        }
    }
    for(int i=0;i<K;i++)
    {
        if(cluster_size[i]>0)
        {
            centroids[index(i,0,2)]/=cluster_size[i];
            centroids[index(i,1,2)]/=cluster_size[i];
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
    if(argc==3)
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
    bool change_flag=true;
    iterations=0;
    while(change_flag)
    {
        change_flag=assign_points_to_clusters();
        calculate_new_centroids();
        iterations++;
    }
    printResults(output_filename);
    free(points);
    free(centroids);
    free(cluster_size);
    free(clusters);
    return 0;
}