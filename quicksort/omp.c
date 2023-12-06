#include <stdio.h>
#include <stdlib.h>
#include "omp.h"


void partition(long int *arr, long int left, long int right)
{
    long int mid=(right + left)/2;
    long int pivotInd = arr[mid];
    long int *leftPtr = arr + left;
    long int *rightPtr = arr + right;
    
    while (leftPtr<=rightPtr)
    {
        long int lVal = *leftPtr;
        long int rVal = *rightPtr;

        for (; lVal < pivotInd && leftPtr<arr + right;)
        {
            leftPtr++;
            lVal = *leftPtr;
        }

        for (; rVal>pivotInd && rightPtr>arr+left;)
        {
            rightPtr--;
            rVal = *rightPtr;
        }

        if (leftPtr <= rightPtr)
        {
            *leftPtr = rVal;
            *rightPtr = lVal;
            leftPtr++;
            rightPtr--;
        }
    }

    long int currRight=rightPtr-arr;
    long int currLeft=leftPtr-arr;

    if (left < currRight)   /* new block is launched to sort the left subaraay */
    {
#pragma omp task
        {partition(arr, left, currRight);}
    }

    if (currLeft < right)   /* same for right - new block launch */
    {
#pragma omp task
        {partition(arr, currLeft, right);}
    }
}
void quicksort(long int *arr, long int n)
{
    int num_threads = 16;
    omp_set_num_threads(num_threads);
    #pragma omp target data map(tofrom : arr) map(to : n)
    #pragma omp parallel
    {
        #pragma omp single nowait
        {partition(arr, 0, n - 1);}
    }
}

void writeToFile(long int *arr, long n, FILE *fw)
{

    for(long int i=0; i<n ;i++)
    {
	    fprintf(fw,"%ld\n",arr[i]);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("usage: Executable name\n");
        printf("name = The name of the input file\n");
	    printf("name = The name of the output file\n");
        exit(1);
    }

    long int n;
    FILE *fp;

    fp = fopen(argv[1], "r");   /*input file */
    if(fp == NULL)
    {
        printf("File does not exist \n");
        exit(1);
    }

    fscanf(fp, "%ld", &n);  /* number of elements */
    size_t size = n * sizeof(long int);
    long int *input = (long int*)malloc(size);

    for(long int i=0;i<n;i++)
    {
        fscanf(fp, "%ld", &input[i]);
    }

    quicksort(input,n);
    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(input, n, fw);
    return 0;
}