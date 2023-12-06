#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>



// Non recursive sorting happens when recursion depth is reached
__device__ void non_recursive(long int *arr, long int left, long int right)
{
    for (long int i = left; i <= right; ++i)
    {
        long int min_val = arr[i];
        long int min_idx = i;
        long int j = i + 1;

        while(j <= right)
        {
            long int val_j = arr[j];
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
            arr[min_idx] = arr[i];
            arr[i] = min_val;
        }
    }
}

// Kernel function
__global__ void kernel_quicksort( long int *arr, long int left, long int right, long int depth)
{
    if (depth >= 20 || right - left <= 40)
    {
        non_recursive(arr, left, right);
        return;
    }

    long int mid = (right + left)/2;
    long int pivotInd = arr[mid];
    long int *leftPtr = arr + left;
    long int *rightPtr = arr + right;

    cudaStream_t streamLeft, streamRight;

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

    long int currRight = rightPtr - arr;
    long int currLeft = leftPtr - arr;

    if (left < currRight)   /* new block is launched to sort the left subaraay */
    {
        cudaStreamCreateWithFlags(&streamLeft, cudaStreamNonBlocking);
        kernel_quicksort<<<1, 1, 0, streamLeft>>>(arr, left, currRight, depth + 1);
        cudaStreamDestroy(streamLeft);
    }

    if (currLeft < right)   /* same for right - new block launch */
    {
        cudaStreamCreateWithFlags(&streamRight, cudaStreamNonBlocking);
        kernel_quicksort<<<1, 1, 0, streamRight>>>(arr, currLeft, right, depth + 1);
        cudaStreamDestroy(streamRight);
    }
}

void quicksort(long int *arr_device, long int n)
{

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 20);   /* recursion depth = 20 */

    int number_blocks = 1;
    int threads_per_block = 1;
    long int left = 0;
    long int right = n - 1;
    
    kernel_quicksort<<<number_blocks, threads_per_block>>>(arr_device, left, right, 0); /* parallel code executed here */

    cudaDeviceSynchronize();
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

    fscanf(fp, "%ld", &n);   /* number of elements */

    size_t size = n*sizeof(long int);
    long int *input = (long int *)malloc(size);

    for(long int i=0;i<n;i++)
    {
        fscanf(fp, "%ld", &input[i]);
    }

    long int *arr_device;
    cudaMalloc((void **)&arr_device, size);
    if (!arr_device)
    {
        printf("Error allocating array arr_device\n");
        exit(1);
    }

    cudaMemcpy(arr_device, input, size, cudaMemcpyHostToDevice);    /* Copy H to D */
    quicksort(arr_device, n);
    cudaMemcpy(input, arr_device, size, cudaMemcpyDeviceToHost);    /* Copy D tp H */

    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(input, n, fw);

    return 0;
}