#include<string.h>
#include<stdio.h>
#include<stdlib.h>

void print_ans(int *arr, long long n, FILE *fw)
{
    long long int i = 0;
    while(i < n)
    {
        printf("%d ", arr[i]);
	fprintf(fw,"%d\n",arr[i]);
        i++;
    }
    
    printf("\n");
}

void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

int partition(int* arr, int left, int right) {
  
  int index = left + ((right - left) / 2);
  swap(&arr[index], &arr[right]);
  int pivot = arr[right];

  int i = (left - 1);
  int j = left;

 while (j < right) {
    if (arr[j] <= pivot) {
      i++;
      swap(&arr[i], &arr[j]);
    }
    j++;
  }

int partition_index = i + 1;
  swap(&arr[partition_index], &arr[right]);

return partition_index;
}


void quicksort_serial(int arr[], int left, int right) {
  if (left < right) {
    int pivot = partition(arr, left, right);
    quicksort_serial(arr, left, pivot - 1);
    quicksort_serial(arr, pivot + 1, right);
  }
}


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("usage: ./quicksort_serial name\n");
        printf("name = The name of the input file\n");
	printf("name = The name of the output file\n");
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

    quicksort_serial(input, 0, n-1);
    FILE *fw;
    fw = fopen(argv[2], "a");
    print_ans(input, n, fw);
    //fw.close();
    return 0;
}
