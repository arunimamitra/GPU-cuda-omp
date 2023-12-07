#include <random>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
#include<ctime>

using namespace std;
void generate_input(string filename,int N) 
{
    srand(static_cast<unsigned int>(std::time(nullptr)));
    ofstream outputFile(filename.c_str());
    if (outputFile.is_open()) 
    {
        outputFile << N << "\n";
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
                outputFile << rand() % 100 << " "; // Generating random numbers between 0 and 99
            outputFile << "\n";
        }
        outputFile.flush();
        outputFile.close();
    }
}

int main(int argc, char **argv) 
{
    if (argc != 3) 
    {
        fprintf(stderr, "usage: gen_conv N filename\n");
        fprintf(stderr, "N= Size of square matrix\n");
        fprintf(stderr, "filename= path to file \n");
        exit(1);
    }
    int N=stoi(argv[1]);
    string input_filename=argv[2];
    generate_input(input_filename,N);
    return 0;
}
