import numpy as np

def read_matrix(filename):
    with open(filename) as f:
        n = int(f.readline())
        data = []

        for _ in range(n):
            data.append(list(map(float, f.readline().split())))

    return np.array(data)

A = read_matrix(r"D:\пп\lab2_2\main\matrixA.txt")
B = read_matrix(r"D:\пп\lab2_2\main\matrixB.txt")
matrix_cpp = read_matrix(r"D:\пп\lab2_2\main\result.txt")

matrix_python = A @ B

if np.allclose(matrix_cpp, matrix_python):
    print("Verification SUCCESS")
else:
    print("Verification FAILED")