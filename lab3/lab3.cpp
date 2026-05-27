#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>

using namespace std;

vector<vector<double>> readMatrix(string filename, int& n)
{
    ifstream file(filename);
    if (!file.is_open())
    {
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
    file.close();
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    vector<vector<double>> A, B;

    double start_time, end_time;

    if (rank == 0)
    {
        int n1, n2;
        A = readMatrix("matrixA.txt", n1);
        B = readMatrix("matrixB.txt", n2);

        if (n1 != n2)
        {
            cout << "Matrix sizes are different" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = n1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        A.resize(n, vector<double>(n));
        B.resize(n, vector<double>(n));
    }

    for (int i = 0; i < n; i++)
    {
        MPI_Bcast(B[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < n; i++)
    {
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));

    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[start_row + i][k] * B[k][j];
            }
            local_C[i][j] = sum;
        }
    }

    vector<vector<double>> C;

    if (rank == 0)
    {
        C.resize(n, vector<double>(n));

        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < n; j++)
                C[start_row + i][j] = local_C[i][j];

        for (int p = 1; p < size; p++)
        {
            int p_start_row = p * rows_per_proc + min(p, remainder);
            int p_local_rows = rows_per_proc + (p < remainder ? 1 : 0);

            for (int i = 0; i < p_local_rows; i++)
            {
                vector<double> row(n);
                MPI_Recv(row.data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < n; j++)
                    C[p_start_row + i][j] = row[j];
            }
        }
    }
    else
    {
        for (int i = 0; i < local_rows; i++)
        {
            MPI_Send(local_C[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    end_time = MPI_Wtime();

    if (rank == 0)
    {
        writeMatrix("result.txt", C);

        long long operations = 2LL * n * n * n;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Number of processes: " << size << endl;
        cout << "Operations: " << operations << endl;
        cout << "Execution time: " << (end_time - start_time) << " seconds" << endl;
        cout << "Performance: " << (operations / (end_time - start_time) / 1e9) << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}