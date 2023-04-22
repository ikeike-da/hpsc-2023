#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
#pragma omp parallel for shared(key)
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  std::vector<int> offset_tmp(range, 0);
#pragma omp parallel for shared(offset)
  for (int i=1; i<range;i++)
      offset[i] = bucket[i-1];
#pragma omp parallel shared(offset)
  for (int j=1;j<range;j<<=1) {
#pragma omp for
    for (int i=0;i<range;i++) {
      offset_tmp[i] = offset[i];
    }
#pragma omp for
    for (int i=j;i<range;i++)
      offset[i] += offset_tmp[i-j];
  }
#pragma omp parallel for shared(key)
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
