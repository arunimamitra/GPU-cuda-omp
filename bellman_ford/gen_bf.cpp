#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <string>

#define INF 1000000
using namespace std;
int max_dis=10;

void generate_input(string filename,int N)
{

	ofstream outputFile(filename.c_str());
    if (outputFile.is_open()) 
    {
        outputFile << N << "\n";
		for(int i=0;i<N;i++)
        {
            for(int j=0;j<N;j++)
            {
				if(i==j)
					outputFile << 0 << " ";
				else
                	outputFile << rand() % max_dis + 1 << " ";
            }
            outputFile << "\n";
        }
        outputFile.flush();
        outputFile.close();
    }
	return;
}

int main(int argc, char *argv[])
{
	if (argc != 3) 
    {
        fprintf(stderr, "usage: gen_bf N filename\n");
        fprintf(stderr, "N= Size of graph matrix\n");
        fprintf(stderr, "filename= path to file \n");
        exit(1);
    }
	int N=stoi(argv[1]);
	string input_filename=argv[2];
    generate_input(input_filename,N);
    return 0;
}