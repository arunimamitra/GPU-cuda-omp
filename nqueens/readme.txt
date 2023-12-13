TO COMPILE :

for sequential version, use :
g++ -std=c++11 -fopenmp  -o sequential sequential.cpp

for openmp version, use :
module load gcc-12.2
gcc -lstdc++ -std=c++11 omp.cpp -fopenmp -lm -o omp

for cuda version, use :
module load gcc-4.9
nvcc -o cuda cuda.cu

TO RUN :
N_max set for N=11

./sequential <number-of-queens>
./omp <number-of-queens>
./cuda <number-of-queens>