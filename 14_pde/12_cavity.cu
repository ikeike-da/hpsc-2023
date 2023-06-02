#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<float>> matrix;

__global__ void calcB(float* b, float* u, float* v, double dt, double dy, double dx, double rho) {
    int ny = gridDim.x;
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    if(j==0||ny-1<=j||i==0||nx-1<=i) return;
    b[j*nx+i] = rho * (1 / dt *
        ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
        ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx))*((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx))
            - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
         (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) -
        ((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy))*((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)));
}

__global__ void copyP(float* p, float* pn) {
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    pn[j*nx+i] = p[j*nx+i];
}

__global__ void calcP(float* p, float* pn, float* b, double dy, double dx) {
    int ny = gridDim.x;
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    if(j==0||ny-1<=j||i==0||nx-1<=i) return;
    p[j*nx+i] = (dy*dy * (pn[j*nx+i+1] + pn[j*nx+i-1]) +
                 dx*dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
                 b[j*nx+i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
}

__global__ void calcBoundaryP(float* p) {
    int ny = gridDim.x;
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    if (j==0)
        p[j*nx+i] = p[(j+1)*nx+i];
    else if (j==ny-1)
        p[j*nx+i] = 0;
    else if (i==0)
        p[j*nx+i] = p[j*nx+i+1];
    else if (i==nx-1)
        p[j*nx+i] = p[j*nx+i-1];
}

__global__ void copyUV(float* u, float* v, float* un, float* vn) {
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    un[j*nx+i] = u[j*nx+i];
    vn[j*nx+i] = v[j*nx+i];
}

__global__ void calcUV(float* u, float* v, float* un, float* vn, float* p, double dt, double dy, double dx, double nu, double rho) {
    int ny = gridDim.x;
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    if(j==0 || ny-1<=j || i==0 || nx-1<=i) return;
    u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i-1])
                           - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])
                           - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                           + nu * dt / dx*dx * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])
                           + nu * dt / dy*dy * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
    v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i-1])
                           - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])
                           - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                           + nu * dt / dx*dx * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])
                           + nu * dt / dy*dy * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
}

__global__ void calcBoundaryUV(float* u, float* v) {
    int ny = gridDim.x;
    int nx = blockDim.x;
    int j = blockIdx.x;
    int i = threadIdx.x;
    if (j==0) {
        u[j*nx+i] = 0;
        v[j*nx+i] = 0;
    }
    else if (j==ny-1) {
        u[j*nx+i] = 1;
        v[j*nx+i] = 0;
    }
    else if (i==0) {
        u[j*nx+i] = 0;
        v[j*nx+i] = 0;
    }
    else if (i==nx-1) {
        u[j*nx+i] = 0;
        v[j*nx+i] = 0;
    }
}


int main(){
    // const int nx = 41;
    // const int ny = 41;
    const int nx = 161;
    const int ny = 161;
    // int nt = 500;
    int nt = 10;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    // double dt = 0.01;
    double dt = 0.001;
    double rho = 1;
    double nu = 0.02;
    float *u, *v, *b, *p, *un, *vn, *pn;
    cudaMallocManaged(&u, ny*nx*sizeof(float));
    cudaMallocManaged(&v, ny*nx*sizeof(float));
    cudaMallocManaged(&b, ny*nx*sizeof(float));
    cudaMallocManaged(&p, ny*nx*sizeof(float));
    cudaMallocManaged(&un, ny*nx*sizeof(float));
    cudaMallocManaged(&vn, ny*nx*sizeof(float));
    cudaMallocManaged(&pn, ny*nx*sizeof(float));
    for (int n=0; n<nt; n++){
        auto tic = chrono::steady_clock::now();
        calcB<<<ny, nx>>>(b, u, v, dt, dy, dx, rho);
        cudaDeviceSynchronize();

        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc - tic).count();
        printf("step=%d: %lf s\n",n,time);
        for (int it=0; it<nit; it++){
            copyP<<<ny, nx>>>(p, pn);
            cudaDeviceSynchronize();
            calcP<<<ny, nx>>>(p, pn, b, dy, dx);
            cudaDeviceSynchronize();
            calcBoundaryP<<<ny, nx>>>(p);
            cudaDeviceSynchronize();
        }

        tic = chrono::steady_clock::now();
        time = chrono::duration<double>(tic-toc).count();
        printf("step=%d: %lf s\n", n, time);
        copyUV<<<ny, nx>>>(u, v, un, vn);
        cudaDeviceSynchronize();
        calcUV<<<ny, nx>>>(u, v, un, vn, p, dt, dy, dx, nu, rho);
        cudaDeviceSynchronize();
        calcBoundaryUV<<<ny, nx>>>(u, v);
        cudaDeviceSynchronize();

        toc = chrono::steady_clock::now();
        time = chrono::duration<double>(toc - tic).count();
        printf("step=%d: %lf s\n",n,time);
    }
}
