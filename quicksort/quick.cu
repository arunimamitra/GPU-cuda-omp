#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to print the numbers in sorted order
void print_ans(int *arr, int n)
{
    register int i = 0;
    while (i < n)
    {
        printf("%d ", arr[i]);
        i++;
    }
    printf("\n");
}

// Selection sort when the max depth is reached
__device__ void selection_sort(int *data, int left, int right)
{
    for (int i = left; i <= right; ++i)
    {
        int min_val = data[i];
        int min_idx = i;
        register int j = i + 1;

        while(j <= right)
        {
            int val_j = data[j];
            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
            ++j;
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

// Kernel function
__global__ void gpu_qsort(int *arr, int left, int right, int depth)
{

    cudaStream_t s_left, s_right;

    if (depth >= 16 || right - left <= 32)
    {
        selection_sort(arr, left, right);
        return;
    }

    int index = left + ((right - left) / 2);
    int pivot = arr[index];
    int *left_ptr = arr + left;
    int *right_ptr = arr + right;
    int left_value, right_value;

    while (left_ptr <= right_ptr)
    {
        left_value = *left_ptr;
        right_value = *right_ptr;

        // Move elements smaller than the pivot value to the left subarray
        for (; left_value < pivot && left_ptr < arr + right;)
        {
            left_ptr++;
            left_value = *left_ptr;
        }

        // Move elements larger than the pivot value to the right subarray
        for (; right_value > pivot && right_ptr > arr + left;)
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

    // Launch a new block to sort the left part.
    if (left < new_right)
    {
        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
        gpu_qsort<<<1, 1, 0, s_left>>>(arr, left, new_right, depth + 1);
        cudaStreamDestroy(s_left);
    }

    // Launch a new block to sort the right part.
    if (new_left < right)
    {
        cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);
        gpu_qsort<<<1, 1, 0, s_right>>>(arr, new_left, right, depth + 1);
        cudaStreamDestroy(s_right);
    }
}

void quick_sort_cuda(int *arr_device, int n)
{

    //Setting the maximum dynamic recursion depth to 16 after which we will use selection sort
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);

    int number_blocks = 1;
    int threads_per_block = 1;
    int left = 0;
    int right = n - 1;
    
    // Calling the kernel function
    gpu_qsort<<<number_blocks, threads_per_block>>>(arr_device, left, right, 0);

    cudaDeviceSynchronize();
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./quicksort_cuda name\n");
        printf("name = The name of the input file\n");
        exit(1);
    }

    int n;
    FILE *fp;

    // Opening the file in the read mode
    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("File does not exist \n");
        exit(1);
    }

    // Getting the number of elements in the file
    fscanf(fp, "%d", &n);

    // Allocating the array to input all the elements from the file
    size_t size = n * sizeof(int);
    int *input = (int *)malloc(size);

    register int i = 0;

    // Populating the array
    while (i < n)
    {
        fscanf(fp, "%d", &input[i]);
        i++;
    }

    int *arr_device;
    cudaMalloc((void **)&arr_device, size);
    if (!arr_device)
    {
        printf("Error allocating array arr_device\n");
        exit(1);
    }

    // Copying the original array to the device
    cudaMemcpy(arr_device, input, size, cudaMemcpyHostToDevice);

    // Core logic.
    quick_sort_cuda(arr_device, n);

    // Copying the answer back to the host
    cudaMemcpy(input, arr_device, size, cudaMemcpyDeviceToHost);

    // Printing the output
    print_ans(input, n);

    return 0;
}
