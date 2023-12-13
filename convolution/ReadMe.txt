Steps to compile and run code-

module load cuda-10.2
module load gcc-4.9.2
 
For Compiling and Executing Sequential Version run-

gcc -o conv_seq conv_seq.cpp -lstdc++ -lm -std=c++11
./conv_seq <N> <output_filename>

For Compiling and executing CUDA run-

nvcc -o conv_cuda conv_cuda.cu -lstdc++ -lm -std=c++11
./conv_cuda <N> <output_filename>

For Compiling and executing OMP run-

module load gcc-12.2
gcc -o conv_omp conv_omp.cpp -lstdc++ -lm -std=c++11 -fopenmp
./conv_omp <N> <output_filename>



