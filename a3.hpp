#ifndef A3_HPP
#define A3_HPP

#include <stdio.h>
#include <math.h>
#include <assert.h>

#define pi 3.14
#define r2pi sqrt(2*pi)

using namespace std;

const int numThreadsPerBlock = 1024;

static void HandleError( cudaError_t err,                         const char *file,                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*
float seq_fhx(std::vector<float> x, int n, float h){

    float sum = 0;
    float fhx_const = 1/(n*h*r2pi);

    for(int i=0;i<n;i++){
        float temp = 0;
        for(int j=0;j<n;j++){
            float t = x[i] - x[j];
            float exponent = exp((-1*t*t)/(2*h*h));
            float k = fhx_const * exponent;
            temp += k;          
        }
        //cout<<temp<<" ";
        sum += temp;        //each temp is a y_i
    }
    //cout<<endl;
    return sum;
}
*/
__global__ void fhx (const float* gin, float* gout, float val, int n, float h) {

    float fhx_const = 1/(n*h*r2pi);
    float k, temp = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float p = gin[tid];

    __shared__ float cache[numThreadsPerBlock];    
    int cacheIndex = threadIdx.x;    
    cache[cacheIndex] = temp; 

    while(tid < n){
        k = (val - p) * (val - p); 
        k = exp(-k/(2*h*h)) * fhx_const;
        temp += k;
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
        cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        gout[blockIdx.x] = cache[0]; 

}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
/*    
    cudaDeviceProp prop;    
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
    printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf( "Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
    printf( "\n" );
*/    
    const int numBlocks = (n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    
    std::vector<float> z(n,0);
    float *dev_x, *dev_z;
    float dev_val, retval;
    
    HANDLE_ERROR(cudaMalloc((void**) &dev_x, n*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_z, n*sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(dev_x, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    for(int i=0; i < n; i++){
        dev_val = x[i];
        fhx <<< numBlocks, numThreadsPerBlock >>> (dev_x, dev_z, dev_val, n, h);

        HANDLE_ERROR(cudaMemcpy(z.data(), dev_z, n*sizeof(float), cudaMemcpyDeviceToHost));
            
        retval = 0;
        for(int k=0;k<numBlocks;k++)
            retval += z[k];
        y[i] = retval;
    }
/*
    float p_sum = 0;
    for(int i=0;i<n;i++)
        p_sum += y[i];    
    float seq_retval = seq_fhx(x, n, h);
    cout<<endl;

    for(int i = 0; i<y.size(); i++){
        //std::cout<<y[i]<<" ";
    }

    printf("\n\nSeq: %f\tParallel: %f\n", seq_retval, p_sum);
    cout<<endl;
    cout<<numBlocks<<endl;
*/
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_z));
} // gaussian_kde

#endif // A3_HPP
