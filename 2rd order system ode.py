# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:42:39 2022

@author: sojorei
"""

#importing important libraries
import numpy as np
import matplotlib.pyplot as plt

##the system of odes function
def func(y, t):
    return np.array([y[1], np.sin(y[0])])

#initial conditions
y0 = np.array([np.pi /4, 0.0])

#n values param
P=int(input('Enter the value of power(p) e.g 2**1 => P=1 P: '))
n=2**P

#set range val
t = np.linspace(0, 10, n)

#runge kutta function
def rungekutta4(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + k1 * h / 2, t[i] + h / 2)
        k3 = f(y[i] + k2 * h / 2, t[i] + h / 2)
        k4 = f(y[i] + k3 * h, t[i] + h)
        y[i+1] = (k1 + 2*k2 + 2*k3 + k4)*(h / 6) +y[i] 
    return y

runge_kutta_solution= rungekutta4(func, y0, t)

plt.plot(t, runge_kutta_solution[:, 0], label=r'$\theta(t)$')
plt.plot(t, runge_kutta_solution[:, 1], label=r"$\theta(t)$'")
plt.ylabel("theta and theta' values" )
plt.xlabel('t vals')
plt.legend()
plt.grid()

#finding the solution at a=30
np.interp(30, t,runge_kutta_solution[:, 0]) #runge_kutta_solution
print(f'The approximation is :  {np.interp(30, t,runge_kutta_solution[:, 0])}')