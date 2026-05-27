import numpy as np

n = 2000 

A = np.random.randint(0, 10, (n, n))
B = np.random.randint(0, 10, (n, n))

def save_matrix(filename, M):
    with open(filename, "w") as f:
        f.write(str(len(M)) + "\n")
        for row in M:
            f.write(" ".join(map(str, row)) + "\n")

save_matrix("matrixA.txt", A)
save_matrix("matrixB.txt", B)

print("Matrices generated")