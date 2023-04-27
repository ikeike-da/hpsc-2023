#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float reductionAdd(int N, __m256 avec){
  float a[N];
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(a, bvec);
  return a[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], is[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    is[i] = i;
  }

  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 isvec = _mm256_load_ps(is);
  for(int i=0; i<N; i++) {
    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }

    __m256 rxvec = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);
    __m256 ryvec = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);
    __m256 rvec = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    __m256 r3vec = _mm256_mul_ps(_mm256_mul_ps(rvec, rvec), rvec);
    __m256 mask = _mm256_cmp_ps(isvec, _mm256_set1_ps(i), _CMP_NEQ_OQ);
    __m256 axvec = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_div_ps(_mm256_mul_ps(rxvec, mvec), r3vec), mask);
    fx[i] -= reductionAdd(N, axvec);
    __m256 ayvec = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_div_ps(_mm256_mul_ps(ryvec, mvec), r3vec), mask);
    fy[i] -= reductionAdd(N, ayvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
