#ifndef CUCALL_H
#define CUCALL_H

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n\t%s: %s\n",__FILE__,__LINE__, cudaGetErrorName(x), cudaGetErrorString(x));\
    exit(1);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n", x, __FILE__,__LINE__);\
    exit(2);}} while(0)

#endif
