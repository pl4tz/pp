#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;


vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);  
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return {};
    }

    file >> n;
    vector<vector<double>> M(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> M[i][j];

    file.close();
    return M;
}


void writeMatrix(const string& filename, vector<vector<double>>& M) {
    ofstream file(filename);
    int n = M.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << M[i][j] << " ";
        file << endl;
    }
    file.close();
}


__global__ void matMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main() {
    int n;
    vector<vector<double>> h_A = readMatrix("matrixA.txt", n);
    vector<vector<double>> h_B = readMatrix("matrixB.txt", n);

    if (h_A.empty() || h_B.empty()) return 1;

    vector<double> A(n*n), B(n*n), C(n*n, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i*n + j] = h_A[i][j];
            B[i*n + j] = h_B[i][j];
        }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n*n*sizeof(double));
    cudaMalloc(&d_B, n*n*sizeof(double));
    cudaMalloc(&d_C, n*n*sizeof(double));

    cudaMemcpy(d_A, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);

    int threads, blockX, blockY;
    cout << "Введите количество потоков: ";
    cin >> threads;
    cout << "Введите размер блока X и Y через пробел: ";
    cin >> blockX >> blockY;

    dim3 block(blockX, blockY);
    dim3 grid((n + block.x - 1)/block.x, (n + block.y - 1)/block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C.data(), d_C, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    vector<vector<double>> h_C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_C[i][j] = C[i*n + j];

    writeMatrix("result.txt", h_C);

    cout << fixed << setprecision(4);
    cout << "Матрица " << n << "x" << n << endl;
    cout << "| Количество потоков | Блок | Время |\n";
    cout << "|:-----------------:|:----:|:-----------:|\n";
    cout << "| " << threads << " | " << blockX << "x" << blockY << " | " << milliseconds << " |" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}