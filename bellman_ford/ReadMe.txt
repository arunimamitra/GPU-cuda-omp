Steps to compile and run code-

module load cuda-10.2
module load gcc-4.9.2

For Compiling and executing input generator run-

gcc -o gen_bf gen_bf.cpp -lstdc++ -lm -std=c++11
./gen_bf <Number_of_vertices> <input_filename_to_store_graph> 

For Compiling and Executing Sequential Version run-

gcc -o bf_seq bf_seq.cpp -lstdc++ -lm -std=c++11
./bf_seq input_100.txt out100_seq

For Compiling and executing CUDA run-

nvcc -o bf_cuda bf_cuda.cu -lstdc++ -lm -std=c++11
./bf_cuda input100.txt out100_cuda

For Compiling and executing OMP run-

module load gcc-12.2
gcc -o bf_omp bf_omp.cpp -lstdc++ -lm -std=c++11 -fopenmp
./bf_omp input100.txt out100_omp




