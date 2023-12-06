#include<string.h>
#include<stdio.h>
#include<stdlib.h>


void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

/*finding pivot index for partitioning array */ 
int partition(int* arr, int left, int right) {
  int index = left + ((right - left) / 2);
  swap(&arr[index], &arr[right]);
  int pivot = arr[right];

  int x = left - 1;
  int y = left;

 while (y < right) {
    if (arr[y] <= pivot) {
      x++;
      swap(&arr[x], &arr[y]);
    }
    y++;
  }

int pivotInd=x + 1;
  swap(&arr[pivotInd], &arr[right]);

return pivotInd;
}

/* main function - recursive call - divide and conquer */
void quicksort(long arr[], int left, int right) {
  if (left < right) {
    int pivot = partition(arr, left, right);
    quicksort(arr, left, pivot - 1);
    quicksort(arr, pivot + 1, right);
  }
}

void writeToFile(long int *arr, long int n, FILE *fw)
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

    quicksort(input, 0, n-1);
    FILE *fw;
    fw = fopen(argv[2], "w");
    writeToFile(input, n, fw);
    return 0;
}
