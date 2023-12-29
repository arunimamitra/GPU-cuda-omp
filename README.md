# CUDA vs OpenMP for GPU Programming
This repository presents a comprehensive comparative analysis of CUDA and OpenMP on the basis of their programmability, scalability, performance and overheads for five diverse and varied algorithms : 
1. Sieve of Eratosthenes
2. Convolution Operation
3. Bellman-Ford Algorithm
4. N-Queens Algorithm
5. Kmeans Clustering

We use NVIDIA's command-line profiler, nvprof, to profile the parallel codes and capture four key metrics -
1. Kernel launch time
2. Total running time
3. Kernal execution time
4. Memory transfer time (between device and host)




