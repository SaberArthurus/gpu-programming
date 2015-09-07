// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ float add (float a, float b)
{
    return a + b;
}


__global__ void add_arrays (float *a, float *b, float *c, int n)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (ind < n)
    {
        c[ind] = add(a[ind], b[ind]);
    }
}


int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    


    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "aux.h"
    
    // initialize the arrays on GPU
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    size_t nbytes = n * sizeof(float);
    cudaMalloc(&d_a, nbytes); CUDA_CHECK;
    cudaMalloc(&d_b, nbytes); CUDA_CHECK;
    cudaMalloc(&d_c, nbytes); CUDA_CHECK;

    // move from host to device memory
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;

    // initialize block and grid size
    dim3 block = dim3(4, 1, 1); 
    dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
    
    // dispatch the kernel 
    add_arrays <<<grid, block>>> (d_a, d_b, d_c, n);

    // copy result back to host memory
    cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free the device memory
    cudaFree(d_a); CUDA_CHECK;
    cudaFree(d_b); CUDA_CHECK;
    cudaFree(d_c); CUDA_CHECK;


    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}



