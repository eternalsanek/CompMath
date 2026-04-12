import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
np.set_printoptions(precision=3, suppress=True)

def lagrange_expr(x_values, y_values):
    x = sp.Symbol('x')
    exp = 0
    n = len(x_values)
    for i in range(n):
        numerator = 1
        denominator = 1
        for j in range(n):
            if i != j:
                numerator *= x - x_values[j]
                denominator *= x_values[i] - x_values[j]
        exp += y_values[i] * numerator / denominator

    return exp

def newton_forward(table_of_difference):
    t = sp.Symbol('t')
    y0 = table_of_difference[1, 0]
    expr = y0
    n = table_of_difference.shape[1]
    for i in range(1, n):
        component = 1
        for j in range(i):
            component *= (t - j)
        component /= math.factorial(i)
        component *= table_of_difference[i + 1, 0]
        expr += component
    return expr

def newton_backward(table_of_difference):
    t = sp.Symbol('t')
    n = table_of_difference.shape[1]
    yn = table_of_difference[1, n - 1]
    expr = yn
    for i in range(1, n):
        component = 1
        for j in range(i):
            component *= (t + j)
        component /= math.factorial(i)
        component *= table_of_difference[i + 1, n - i - 1]
        expr += component
    return expr

def newtons_interpolation(newton_forward, newton_backward, x_values, x, h):
    t = sp.Symbol('t')

    if x < (x_values[len(x_values) - 1] + x_values[0]) / 2:
        temp = (x - x_values[0]) / h
        return newton_forward.subs(t, temp)
    else:
        temp = (x - x_values[len(x_values) - 1]) / h
        return newton_backward.subs(t, temp)

#Задание 1
x_values = np.array([0.68, 0.73, 0.80, 0.88, 0.93, 0.99])
y_values = np.array([0.80866, 0.89492, 1.02964, 1.20966, 1.34087, 1.52368])

x = sp.Symbol('x')
exp = lagrange_expr(x_values, y_values)
print("Задание 1\n")
print("Интерполяционный многочлен Лагранжа  для неравноотстоящих узлов:\n", sp.expand(exp), "\n")

f_lagrange = sp.lambdify(x, exp, "numpy")

x_plot = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 200)
y_plot = f_lagrange(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='Многочлен Лагранжа', color='blue', linewidth=2)
plt.scatter(x_values, y_values, color='red', label='Узлы интерполяции', zorder=5)

plt.title('Интерполяция с помощью интерполяционного многочлена Лагранжа')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.savefig('lagrange_interpolation.png', dpi=300, bbox_inches='tight')

points_to_calc = np.array([0.896, 0.812, 0.774, 0.955, 0.715])
print("Вычисленные значения функции:")
for i in points_to_calc:
    print(f"F({i}) = {f_lagrange(i)}")
print("\n")

#plt.show()

#Задание 2
x_values = np.array([0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56])
y_values = np.array([20.1946, 19.6133, 18.9425, 18.1746, 17.3010, 16.3123, 15.1984, 13.9484, 12.5508, 10.9937, 9.2647, 7.3510])

points_to_calc = np.array([0.455, 0.5575, 0.44, 0.5674])

table_of_difference = np.zeros((len(x_values) + 1, len(y_values)))
table_of_difference[0] = x_values
table_of_difference[1] = y_values

for i in range(2, len(x_values) + 1):
    for j in range(len(y_values) - i + 1):
        table_of_difference[i, j] = table_of_difference[i - 1, j + 1] - table_of_difference[i - 1, j]

interpolation_forward = newton_forward(table_of_difference)
interpolation_backward = newton_backward(table_of_difference)

print("Задание 2\n")
for i in points_to_calc:
    print(f"F({i}) = {newtons_interpolation(interpolation_forward, interpolation_backward, x_values, i, 0.01)}")