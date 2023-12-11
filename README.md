# GPU-cuda-omp

SIEVE SEQUENTIAL

[am13018@access1 v2]$ ./sequential 100000000 out.txt
Time taken = 2.100000

[am13018@access1 v2]$ ./sequential 500000000 out.txt
Time taken = 11.620000

[am13018@access1 v2]$ ./sequential 10000000 out.txt
Time taken = 0.160000

[am13018@access1 v2]$ ./sequential 50000000 out.txt
Time taken = 1.140000

[am13018@access1 v2]$ ./sequential 1000000 out.txt
Time taken = 0.030000

[am13018@access1 v2]$ ./sequential 5000000 out.txt
Time taken = 0.130000

[am13018@access1 v2]$ ./sequential 500000 out.txt
Time taken = 0.010000

[am13018@access1 v2]$ ./sequential 100000 out.txt
Time taken = 0.000000

[am13018@access1 v2]$ ./sequential 1000000000 out.txt
Time taken = 24.630000



**************


[am13018@cuda3 v2]$  ./sequential 1000000000 out2.txt
Time taken = 46.610000

[am13018@cuda3 v2]$  ./sequential 500000000 out2.txt
Time taken = 21.960000

[am13018@cuda3 v2]$  ./sequential 100000000 out2.txt
Time taken = 3.640000

[am13018@cuda3 v2]$  ./sequential 50000000 out2.txt
Time taken = 1.520000

[am13018@cuda3 v2]$  ./sequential 10000000 out2.txt
Time taken = 0.170000

[am13018@cuda3 v2]$  ./sequential 5000000 out2.txt
Time taken = 0.100000

[am13018@cuda3 v2]$  ./sequential 1000000 out2.txt
Time taken = 0.020000



[am13018@cuda3 v2]$ ./serial 6

Time used = 0.0202285 sec

Result= 4


[am13018@cuda3 v2]$ ./serial 10

Time used = 2622.84 sec

Result= 724


[am13018@cuda3 v2]$ 

[am13018@cuda3 v2]$ 

[am13018@cuda3 v2]$ ./serial 7

Time used = 0.227732 sec

Result= 40


[am13018@cuda3 v2]$ ./serial 8

Time used = 3.7057 sec

Result= 92


[am13018@cuda3 v2]$ ./serial 9

Time used = 91.3039 sec

Result= 352


[am13018@cuda3 v2]$  ./sequential 100000 out2.txt
Time taken = 0.000000

[am13018@cuda3 v2]$  ./sequential 500000 out2.txt
Time taken = 0.010000
