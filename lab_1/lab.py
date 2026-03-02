import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
from sympy import *
from scipy import integrate
from scipy.optimize import fsolve

#Функция для интеграла Scipy
def f(x):
    return 1/(x**2 + 4*x + 9)

#Функция для интеграла Scipy
def g(x):
    return np.e ** (2*x) * np.cos(x)

def f1(x):
    return np.log(x) + 2

def f2(x):
    return -3 * x

def difference(x):
    return f1(x) - f2(x)

#Функция для нахождения всех корней на заданном отрезке [start, end]
def find_all_roots(function, start, end, num = 1000):
    x_values = np.linspace(start, end, num)
    y_values = function(x_values)

    all_roots = []

    for i in range(len(x_values) - 1):
        if (y_values[i] * y_values[i + 1] < 0):
            root = fsolve(function, x_values[i])[0]

            # Проверка на дубликаты
            same_root = False
            for root_exist in all_roots:
                if abs(root - root_exist) < 1e-9:
                    same_root = True
                    break

            if not same_root and start <= root <= end:
                all_roots.append(root)

    return all_roots

#1
print('Задание 1')

numbers = np.random.uniform(-3 + 1e-15, 3, (5, 5))
numbers.transpose()
determinant = np.linalg.det(numbers)
print("Определитель вещественной транспонированной матрицы")
print(determinant, '\n\n')

#2
print('Задание 2')

n = np.array([[1], [2], [3]])
A = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
m = np.dot(A, n)

print(m, '\n\n')

#3
print('Задание 3')

x, y = symbols('x, y')
expr = (7*x*y)/4 * (x + y) - (x - y)**2
print("=============Ваше выражение===========")
pprint(expr)
print("==========Упрощённое выражение========")
new_expr = simplify(expr)
pprint(new_expr)
print("===============Результат==============")

val_x = -1.23
val_y = math.sqrt(8)
res = new_expr.subs({x: val_x, y: val_y})
print(res, '\n\n')

#4
print('Задание 4')
x = Symbol('x')
y = Symbol('y')
expression = 7*x*y*(x+y)/4 - (x-y)**2
diff1 = diff(expression, x)
diff2 = diff(expression, y)

print("=============Ваше выражение===========")
pprint(expression)
print("=======Частная произвдодная по x======")
pprint(simplify(diff1))
print("=======Частная произвдодная по y======")
pprint(simplify(diff2))
print('\n\n')

#5
print('Задание 5')

a = np.array([[-7, -5, -5], [0, 3, 0], [10, 5, 8]], int)
vals, vecs = np.linalg.eig(a)
print("Собственные значения")
print(vals)
print("Собственные вектора")
print(vecs, '\n\n')

#6
print('Задание 6')

quadSciPy, error = integrate.quad(g, 0, np.pi/2)
print('Значение интеграла, вычисленного с использованием SciPy:')
print(quadSciPy)

x = Symbol('x')
function = np.e ** (2 * x) * cos(x)
quadSymPy = sp.integrate(function, (x, 0, np.pi/2))
print('Значение интеграла, вычисленного с использованием SymPy:')
print(quadSymPy, '\n\n')


#7
print('Задание 7')

resForIntegrate1, err = integrate.quad(f, -np.inf, np.inf)
print("Вычисленный интеграл при помощи SciPy и погрешность")
print(resForIntegrate1, err)

x = Symbol('x')
exprForIntegrate = 1/(x**2 + 4*x + 9)
resForIntegrate2 = sp.integrate(exprForIntegrate, (x, -sp.oo, sp.oo))
print("Вычисленный интеграл при помощи SymPy символьно и численно")
print(resForIntegrate2)
print(resForIntegrate2.evalf(), '\n\n')

#8
print('Задание 8')

x1 = np.linspace(0.0001, 2.5, 500)
x2 = np.linspace(-2, 2.5, 500)
y1 = f1(x1)
y2 = f2(x2)

figure, axes = plt.subplots(figsize=(7, 5))

axes.spines['left'].set_position('zero')
axes.spines['bottom'].set_position('zero')
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')

axes.set_xlabel('Ось x', loc='right', fontsize=10)
axes.set_ylabel('Ось y', rotation=0, loc='top', fontsize=10)

plt.plot(x1, y1, 'g', label='ln(x) + 2')
plt.plot(x2, y2, 'b', label='-3x')

plt.grid()
plt.title('Система координат', pad=20)

roots = find_all_roots(difference, 0.0001, 2.5, 1000)

for root in roots:
    plt.plot(root, f1(root), 'o', color='darkorange')

    plt.annotate(f'({root:.4f}, {f1(root):.4f})',
                 xy=(root, f1(root)),
                 xytext=(5, -20),
                 textcoords='offset points',
                 color='red', fontsize=10)

axes.legend()

plt.savefig('graph.png')
plt.show()