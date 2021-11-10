# -*- coding: utf-8 -*-
"""The SIRD model.ipynb

"""

#QUESTION 3D
#(i)

import numpy as np 
import matplotlib.pyplot as plt 
import numba 
import time 
start_time = time.clock() 
from sympy import *
init_printing () 
# Equations:  
N=5000000
beta=0.4
gamma=2.1
alpha=0.2

@numba.jit() 
def V(y,t):
    S=y[0]
    I=y[1]
    R=y[2]
    D=y[3]     
    dSdt=-((beta*I*S )/N)
    dIdt=((beta*I*S )/N)-(gamma*I-alpha*I)
    dRdt=gamma*I
    dDdt=alpha*I   
    return np.array([dSdt,dIdt,dRdt,dDdt])  

a21 , a31 , a32 , a41 , a42 , a43 = symbols ('a21 , a31 , a32 , a41 , a42 , a43 ')
b1 , b2 , b3 , b4 = symbols ('b1 , b2 , b3 , b4 ')
c2 , c3 , c4 = symbols ('c2 , c3 , c4 ')
# Choose 4 values for the unkowns
c2 , c3 , c4 , b2 = Rational (1 ,2) , Rational (1 ,2) , 1 , Rational (1 ,3)
# Define order conditions
eqn1 = b1 + Rational (1 ,2) + b3 + b4 - 1
eqn2 = Rational (1 ,4) + b3 * c3 + b4 * c4 - Rational (1 ,2)
eqn3 = Rational (1 ,8) + b3 * c3 ** 2 + b4 * c4 ** 2 - Rational (1 ,3)
eqn4 = Rational (1 ,16) + b3 * c3 ** 3 + b4 * c4 ** 4 - Rational (1 ,4)
eqn5 = b3 * c3 * a32 * Rational (1 ,2)+ b4 * c4 * ( a42 * Rational (1 ,2) + a43 * c3 ) - Rational (1 ,8)
eqn6 = b3 * a32 + b4 * a42 - b2 * (1 - Rational (1 ,2) )
eqn7 = b4 * a43 - b3 * (1 - c3 )
eqn8 = b4 * (1 - c4 )
eqn9 = Rational (1 ,2) - a21
eqn10 = Rational (1 ,2) - a31 - a32
eqn11 = c4 - a41 - a42 - a43
@numba.jit() 
def eRK(f, u0, t0, tf , n):     
  t = np.linspace(t0, tf, n+1)     
  u = np.array((n+1)*[u0])
  h = t[1]-t[0]     
  for i in range(n):         
    k1 = h * f(u[i], t[i])             
    k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)         
    k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)         
    k4 = h * f(u[i] + k3, t[i] + h)         
    u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6     
  return u, t  
u, t  = eRK(V, np.array([10.,20.,10.,20.]) , 0. , 100. , 2000000) 

x,vx,y,vy= u.T 

plt.plot(t, x, t, y) 
plt.plot(x,y) 
plt.grid('on') 
plt.legend(['x(t)','y(t)']) 
plt.legend(['y(x)']) 
plt.show()

#(ii)

#from the plot above we can estimate the beta target which occurs at the intersection with beta target as 33/100=0.33 whilest the peak day at 
#which infection occurs is approximated as 4th day
#we gonna use the same program buh the rate of infecction and target infection is changed as follows
#for a population of no more than 200000 we will have a beta rate of 0.33,gamma rate of 0.0105,apha=0.003

import numpy as np 
import matplotlib.pyplot as plt 
import numba 
import time 
start_time = time.clock() 
from sympy import *
init_printing () 
# Equations:  
N=5000000
beta=0.4
beta_target=0.33
gamma=0.0105
alpha=0.003

@numba.jit() 
def V(y,t):
    S=y[0]
    I=y[1]
    R=y[2]
    D=y[3]     
    dSdt=-((beta_target*I*S )/N)
    dIdt=((beta_target*I*S )/N)-(gamma*I-alpha*I)
    dRdt=gamma*I
    dDdt=alpha*I   
    return np.array([dSdt,dIdt,dRdt,dDdt])  

a21 , a31 , a32 , a41 , a42 , a43 = symbols ('a21 , a31 , a32 , a41 , a42 , a43 ')
b1 , b2 , b3 , b4 = symbols ('b1 , b2 , b3 , b4 ')
c2 , c3 , c4 = symbols ('c2 , c3 , c4 ')
# Choose 4 values for the unkowns
c2 , c3 , c4 , b2 = Rational (1 ,2) , Rational (1 ,2) , 1 , Rational (1 ,3)
# Define order conditions
eqn1 = b1 + Rational (1 ,2) + b3 + b4 - 1
eqn2 = Rational (1 ,4) + b3 * c3 + b4 * c4 - Rational (1 ,2)
eqn3 = Rational (1 ,8) + b3 * c3 ** 2 + b4 * c4 ** 2 - Rational (1 ,3)
eqn4 = Rational (1 ,16) + b3 * c3 ** 3 + b4 * c4 ** 4 - Rational (1 ,4)
eqn5 = b3 * c3 * a32 * Rational (1 ,2)+ b4 * c4 * ( a42 * Rational (1 ,2) + a43 * c3 ) - Rational (1 ,8)
eqn6 = b3 * a32 + b4 * a42 - b2 * (1 - Rational (1 ,2) )
eqn7 = b4 * a43 - b3 * (1 - c3 )
eqn8 = b4 * (1 - c4 )
eqn9 = Rational (1 ,2) - a21
eqn10 = Rational (1 ,2) - a31 - a32
eqn11 = c4 - a41 - a42 - a43
@numba.jit() 
def eRK(f, u0, t0, tf , n):     
  t = np.linspace(t0, tf, n+1)     
  u = np.array((n+1)*[u0])
  h = t[1]-t[0]     
  for i in range(n):         
    k1 = h * f(u[i], t[i])             
    k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)         
    k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)         
    k4 = h * f(u[i] + k3, t[i] + h)         
    u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6     
  return u, t  
u, t  = eRK(V, np.array([10.,20.,10.,20.]) , 0. , 100. , 2000000) 

x,vx,y,vy= u.T 

plt.plot(t, x, t, y) 
plt.plot(x,y) 
plt.grid('on')  
plt.legend(['x(t)','y(t)_beta_target','y(x)beta04']) 
plt.show()

