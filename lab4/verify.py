import numpy as np

def read_matrix(filename):
    with open(filename) as f:
        n = int(f.readline())
        data = []

        for _ in range(n):
            data.append(list(map(float, f.readline().split())))

    return np.array(data)

A = read_matrix("matrixA.txt")
B = read_matrix("matrixB.txt")
matrix_cpp = read_matrix("result.txt")

matrix_python = A @ B

if np.allclose(matrix_cpp, matrix_python):
    print("Verification SUCCESS")
else:
    print("Verification FAILED")