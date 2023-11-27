#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

void print_ans(int *arr, int n)
{
    register int i = 0;
    while(i < n)
    {
        printf("%d ", arr[i]);
        i++;
    }
    printf("\n");
}

void helper(int *arr, int left, int right)
{
    int index = left + ((right - left) / 2);
    int pivot = arr[index];
    int left_value;
    int right_value;
    int *left_ptr = arr + left;
    int *right_ptr = arr + right;
    
    while (left_ptr <= right_ptr)
    {
        left_value = *left_ptr;
        right_value = *right_ptr;

	for(;left_value < pivot && left_ptr < arr + right;)
        {
            left_ptr++;
            left_value = *left_ptr;
        }
	for(;right_value > pivot && right_ptr > arr + left;)
        {
            right_ptr--;
            right_value = *right_ptr;
        }

        if (left_ptr <= right_ptr)
        {
            *left_ptr = right_value;
            *right_ptr = left_value;
            left_ptr++;
            right_ptr--;
        }
    }

    int new_right = right_ptr - arr;
    int new_left = left_ptr - arr;

    if (left < new_right)
    {
#pragma omp task
        {
            helper(arr, left, new_right);
        }
    }

    if (new_left < right)
    {
#pragma omp task
        {
            helper(arr, new_left, right);
        }
    }
}
void omp_quick_sort(int *arr, int n)
{
    int num_threads = 10;
    omp_set_num_threads(num_threads);
#pragma omp target data map(tofrom : arr) map(to : n)
#pragma omp parallel
    {
#pragma omp single nowait
        {
            helper(arr, 0, n - 1);
        }
    }
}

int main(int argc, char *argv[])
{

	if (argc != 2)
    {
        printf("usage: ./quicksort_openmp name\n");
        printf("name = The name of the input file\n");
        exit(1);
    }

    int n;
    FILE *fp;

    fp = fopen(argv[1], "r");
    if(fp == NULL)
    {
        printf("File does not exist \n");
        exit(1);
    }

    fscanf(fp, "%d", &n);

    size_t size = n * sizeof(int);
    int *input = (int *)malloc(size);

    register int i = 0;

    while(i < n)
    {
        fscanf(fp, "%d", &input[i]);
        i++;
    }

    omp_quick_sort(input, n);
    print_ans(input, n);

    return 0;
}
