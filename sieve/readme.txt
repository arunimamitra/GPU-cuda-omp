TO COMPILE :

for sequential version, use:
gcc -std=c99 -lm sequential.c -o sequential

for openmp version, use :
module load gcc-12.2
gcc omp.cpp -lstdc++ -lm -fopenmp -std=c++11 -o omp

for cuda version, use :
module load gcc-4.9
nvcc -o cudaF cuda.cu

TO RUN :

./sequential <N> <output-file-name>
./omp <N> <output-file-name>
./cuda <N> <output-file-name>