import numpy as np

#Задание 1
rng = np.random.default_rng()
A = rng.uniform(2, 4, (10, 10))

a = A[3, :]
b = A[:, 4]

print("Задание 1")
print("Матрица A:\n", A)
print("Скалярное произведение: ", np.dot(a, b), "\n")

#Задание 2
A = np.random.randint(2, 7, (6, 4))
B = np.random.randint(2, 7, (4, 5))
C1 = np.zeros((6,5), int)
C2 = np.zeros((6,5), int)
C3 = np.dot(A, B)

#1 способ
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        for k in range(A.shape[1]):
            C1[i, j] += A[i, k] * B[k, j]

#2 способ
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        C2[i, j] = np.dot(A[i, :], B[:, j])

print("Задание 2")
print("Матрица C1:\n", C1)
print("Матрица C2:\n", C2)
print("Матрица C3:\n", C3, "\n")

#Задание 3
A = np.random.randint(1, 11, (5, 5))
A_Tril = np.tril(A, 0)
B = np.random.randint(1, 100, 5)

X = np.zeros(5)

X[0] = B[0] / A_Tril[0, 0]
for i in range(1, 5):
    X[i] = ( B[i] - np.dot(A_Tril[i, 0:i], X[0:i]) ) / A_Tril[i, i]

X_Lib = np.linalg.solve(A_Tril, B)

print("Задание 3")
print("Самостоятельно вычисленный вектор:\n", X, "\n")
print("Вектор, вычисленный с помощью solve:\n", X_Lib, "\n")

#Задание 4
A = np.array([[8.2, -3.2, 14.2, 14.8],
               [5.6, -12, 15, -6.4],
               [5.7, 3.6, -12.4, -2.3],
               [6.8, 13.2, -6.3, -8.7]])

B = np.array([-8.4, 4.5, 3.3, 14.3])

U = np.zeros((4, 4), float)
L = np.identity(4, float)

for i in range(4):
    for j in range(4):
        if i <= j:
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        if i > j:
            L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

print("Задание 4")
print("L:\n", L, "\n")
print("U:\n", U, "\n")
A_check = np.dot(L, U)
print("Исходная матрица:\n", A, "\n")
print("Произведение L и U:\n", A_check, "\n")

Y = np.zeros(4, float)
Y[0] = B[0]
for i in range(1, 4):
    Y[i] = B[i] - np.dot(L[i, 0:i], Y[0:i])

X = np.zeros(4, float)
X[3] = Y[3] / U[3, 3]
for i in range(2, -1, -1):
    X[i] = (Y[i] - np.dot(U[i, i + 1:], X[i + 1:])) / U[i, i]

X_Lib = np.linalg.solve(A, B)
print("Самостоятельно вычисленный вектор:\n", X, "\n")
print("Вектор, вычисленный с помощью solve:\n", X_Lib, "\n")