Steps to compile and run code-

module load cuda-10.2
module load gcc-4.9.2

For Compiling and executing input generator run-

gcc -o gen_kmeans gen_kmeans.cpp -lstdc++ -lm -std=c++11
./gen_kmeans <Number_of_coordinates> <input_filename_to_store_graph> 

For Compiling and Executing Sequential Version run-

gcc -o kmeans_seq kmeans_seq.cpp -lstdc++ -lm -std=c++11
./kmenas_seq input_100.txt out100_seq

For Compiling and executing CUDA run-

nvcc -o kmeans_cuda kmeans_cuda.cu -lstdc++ -lm -std=c++11
./kmeans_cuda input100.txt out100_cuda

For Compiling and executing OMP run-

module load gcc-12.2
gcc -o kmeans_omp kmeans_omp.cpp -lstdc++ -lm -std=c++11 -fopenmp
./kmeans_omp input100.txt out100_omp




