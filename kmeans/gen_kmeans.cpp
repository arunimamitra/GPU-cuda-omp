#include <random>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<iostream>
#include<fstream>
using namespace std;


void generate_input(string filename,int N) 
{
    ofstream outputFile(filename.c_str());
    outputFile << N << "\n";
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
    for (int i = 0; i < N; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        outputFile << x << " " << y << std::endl;
    }
    outputFile.flush();
    outputFile.close();
}

int main(int argc, char **argv) 
{
    if (argc != 3) 
    {
        fprintf(stderr, "usage: gen_kmeans N filename\n");
        fprintf(stderr, "N= no of coordinates to be generated\n");
        fprintf(stderr, "filename= path to file \n");
        exit(1);
    }
    int N=stoi(argv[1]);
    string input_filename=argv[2];
    generate_input(input_filename,N);
    return 0;
}
