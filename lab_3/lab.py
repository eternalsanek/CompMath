import numpy as np
np.set_printoptions(precision=3, suppress=True)

def householder_reflection(X):
    a = X
    E = np.identity(a.shape[0], float)
    e1 = np.zeros((a.shape[0], 1), float)
    e1[0, 0] = 1
    v = a + np.sign(a[0, 0]) * np.linalg.norm(a, axis=0)[0] * e1
    w = v / np.linalg.norm(v, axis=0)[0]
    V = E - 2 * np.dot(w, np.transpose(w))
    return V

def qr_decomposition(A):
    p = []
    R = A.copy()
    for i in range(A.shape[1]):
        a = R[i:, i:i+1]
        P0 = householder_reflection(a)
        P = np.identity(A.shape[0], float)
        P[i:, i:] = P0
        p.append(P)
        R = np.dot(P, R)

    Q = p[len(p) - 1]
    for i in range(len(p) - 2, -1, -1):
        Q = np.dot(Q, p[i])

    return np.transpose(Q), R

#Задание 1
X = np.random.randint(1, 100, (10, 1))
NX_1 = 0
for i in range(X.shape[0]):
    NX_1 += np.abs(X[i][0]) ** 3
NX_1 = NX_1 ** (1/3)
NX_2 = np.linalg.norm(X, ord=3, axis=0)[0]

print("Задание 1\n")
print("Норма вектора, вычисленная самостоятельно: ", NX_1)
print("Норма вектора, вычисленная с помощью linalg.norm: ", NX_2, "\n")

#Задание 2
A = np.random.randint(1, 100, (7, 8))
NA_1 = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        NA_1 += np.abs(A[i][j]) ** 2
NA_1 = NA_1 ** (1/2)
NA_2 = np.linalg.norm(A)

print("Задание 2\n")
print("Норма матрицы Фробениуса, вычисленная самостоятельно: ", NA_1)
print("Норма матрицы Фробениуса, вычисленная с помощью linalg.norm: ", NA_2, "\n")

#Задание 3
V0 = householder_reflection(X[1:])
V = np.identity(X.shape[0], float)
V[1:, 1:] = V0
X_new = np.dot(V, X)
print("Задание 3\n")
print("Старый вектор:\n", np.transpose(X), "\n")
print("Вектор после отражения:\n", np.transpose(X_new), "\n")

#Задание 4
A = np.array([[1.7, -1.8, 1.9, -57.4],
              [1.1, -4.3, 1.5, -1.7],
              [1.2, 1.4, 1.6, 1.8],
              [7.1, -1.3, -4.1, 5.2]])

B = np.array([10, 19, 20, 10])
X_lib = np.linalg.solve(A, B)

Q, R = qr_decomposition(A)

B = np.dot(np.transpose(Q), B)

X = np.zeros(4, float)
X[3] = B[3] / R[3, 3]
for i in range(2, -1, -1):
    X[i] = (B[i] - np.dot(R[i, i + 1:], X[i + 1:])) / R[i, i]

print("Задание 4\n")

print("Матрица Q:\n", Q, "\n")
print("Матрица R:\n", R, "\n")

print("Матрица Q * R:\n", np.dot(Q, R), "\n")
print("Матрица A:\n", A, "\n")

print("Самостоятельно вычисленный вектор:\n", X, "\n")
print("Вектор, вычисленный с помощью solve:\n", X_lib, "\n")