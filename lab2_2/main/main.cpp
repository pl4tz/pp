#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

vector<vector<double>> readMatrix(string filename, int& n)
{
    ifstream file(filename);
    file >> n;

    vector<vector<double>> M(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> M[i][j];

    return M;
}

void writeMatrix(string filename, vector<vector<double>>& M)
{
    ofstream file(filename);

    int n = M.size();

    file << n << endl;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << M[i][j] << " ";
        file << endl;
    }
}

vector<vector<double>> multiply(vector<vector<double>>& A,
    vector<vector<double>>& B,
    int threads)
{
    int n = A.size();

    vector<vector<double>> C(n, vector<double>(n, 0));

    omp_set_num_threads(threads);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main(int argc, char* argv[])
{
    int threads = 8;

    if (argc > 1)
        threads = atoi(argv[1]);

    int n1, n2;

    auto A = readMatrix("D:\\пп\\lab2_2\\main\\matrixA.txt", n1);
    auto B = readMatrix("D:\\пп\\lab2_2\\main\\matrixB.txt", n2);

    if (n1 != n2)
    {
        cout << "Matrix sizes are different" << endl;
        return 1;
    }

    int n = n1;

    auto start = chrono::high_resolution_clock::now();

    auto C = multiply(A, B, threads);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> time = end - start;

    writeMatrix("D:\\пп\\lab2_2\\main\\result.txt", C);

    long long operations = 2LL * n * n * n;

    cout << "Threads: " << threads << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Operations: " << operations << endl;
    cout << "Execution time: " << time.count() << " seconds" << endl;

    return 0;
}