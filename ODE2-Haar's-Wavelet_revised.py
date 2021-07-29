# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 10:33:57 2021

@author: sojore
"""

import pywt
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# **ODE Equation
# d^4y/dt^4 + 4y=cos2t or y"" + 4y=cos2t
# y(0)=y"(0)=0 ,y(1)=y"(1)=0

# solution:::
#   re-writting the ODE as a  system:
# A=dy/dt
# B=d^2y/dt^2=dA/dt
# C=d^3y/dt^3=dB/dt
# D=d^4y/dt^4=dC/dt
# dD/dt=-4A+cost(2t) 


#define the function

# # YOU CAN CHOOSE EITHER OF THE BELOW METHODS TO SOLVE THE ODE,
# # REMEMBER TO COMMENT OUT ONE METHOD ..

# # 1st MEthod, we use solve_ivp  function to solve our ODE as below
def ode_func_using_solve_ivp(t,y):
    A=y[0]
    B=y[1]
    C=y[2]
    D=y[3]
      
    #derivertives
    dAdt=B
    dBdt=C
    dCdt=D
    dDdt=-4*A+math.cos(2*t)
    
    return np.array([dAdt,dBdt,dCdt,dDdt])
    
t_span=np.array([0,50])
times=np.linspace(t_span[0],t_span[1],100)

#set initial conditions
y0=np.array([0,1,0,1])

#solving the ODE using solve_ivp function
solution=solve_ivp(ode_func_using_solve_ivp,t_span,y0,t_eval=times)
t=solution.t
A=solution.y[0]
B=solution.y[1]
C=solution.y[2]
D=solution.y[3]


# # 2rd Method,we can use ODEINT function rather than solve_ivp to get the same output as below

# def ode_func_using_odeint(y,t):
#     A=y[0]
#     B=y[1]
#     C=y[2]
#     D=y[3]
    
#     #derivertives
#     dAdt=B
#     dBdt=C
#     dCdt=D
#     dDdt=-4*A+math.cos(2*t)
    
#     return np.array([dAdt,dBdt,dCdt,dDdt])
    
# #initializing any initial conditions t0 test thefunction

# y0=[0,1,0,1]#you can set any initial values of your preference

# t=np.linspace(0,50)
# #solving the ODE using ODEINT function
# sol=odeint(ode_func_using_odeint,y0,t)
# A=sol[:,0]
# B=sol[:,1]
# C=sol[:,2]
# D=sol[:,3]



# #  GRAPH'S  PLOTTING   STEP
# # NOTE: UNCOMMENT EACH PLOT AT A TIME FOR INDEPENDENT GRAPHS PLOTTING



# print(f'A=={A}')
# print(f'B=={B}')
# print(f'C=={C}')
# print(f'D=={D}')

# # you can plot the function for the 1st,2nd,3rd and 4th order derivertives soln.

# PLOT 1.1
plt.rc('font',size=14)
plt.figure()
plt.plot(t,A, '--',label='A')
plt.plot(t,B, '--', label='B')
plt.plot(t,C, '--', label='C')
plt.plot(t,D,'--',label='D')
plt.title('Y-function in Time (t)')
plt.xlabel('Time')
plt.ylabel('Concentration points')
plt.legend()
plt.show


# # ploting PLOT 1.1 but in a semilogy function we have

# #PLOT 1.2
# plt.semilogy(t,A,'--', label='A')
# plt.semilogy(t,B,'--',label='B')
# plt.semilogy(t,C,'--',label='C')
# plt.semilogy(t,D,'--',label='D')
# plt.ylabel('Concentration points(log scale)')
# plt.title('Y-function in Time (t)')
# plt.xlabel('Time')
# plt.legend()
# plt.show

   
# wavelet transform analysis of our ODE equation using haar wavelet transform
X=[A[:],B[:],C[:],D[:]]
x=[]
for i in X:
    x.append(i)
    
# #using the HAAR WAVELET TRANSFORM to plot the graphs

cA,cD=pywt.dwt(x,'haar') #this is the wavelet transform

y=pywt.idwt(cA,cD,'haar') #for the inverse wavelet transform we have

# # cA --this gives the approximation haar values
# # cD --this  gives the detailed value coeficients (error function)
# # y ---this gives the reconstracted haar signal
# # x----this is the actual coefficients of the ODE function in question
# print(f'Haar approximation vals {cA}')  
# print(f'Detailed val coef (Error) {cD}')
# print(f'Reconstracted haar signal {y}')
# print(f'Exact DE coefficients {x}')


#plotting the wavelet transform signal

# # PLOT 1.3
# plt.plot(cA,color='blue',label='Haar Approximation')
# plt.plot(x,color='red',label='Exact ODE coefficients')
# plt.xlabel('Time')
# plt.ylabel('Exact(x) value')
# plt.title('Exact (x) in Red vs Haar Approximation (cA) in Blue')

# # PLOT 1.4
# plt.plot(cD,color='blue',label='Error')
# plt.plot(y,color='red',label='reconstracted signal')
# plt.xlabel('Time')
# plt.ylabel('Haars Error')
# plt.title('Haars Error (cD) in Blue vs reconstracted signal in Red')





