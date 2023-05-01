#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void initBucket(int* bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=range) return;
  bucket[i] = 0;
}

__global__ void bucketIn(int* bucket, int* key, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void bucketOut(int* bucket, int* key, int range, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  for (int j=0, offset=0; j<range; j++) {
    offset += bucket[j];
    if (i < offset) {
      key[i] = j;
      break;
    }
  }
}

int main() {
  const int n = 50;
  const int range = 5;
  const int M = 1024;
  int* key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  initBucket<<<(range+M-1)/M,M>>>(bucket, range);
  cudaDeviceSynchronize();
  bucketIn<<<(n+M-1)/M,M>>>(bucket, key, n);
  cudaDeviceSynchronize();
  bucketOut<<<(n+M-1)/M,M>>>(bucket, key, range, n);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
