QUESTION 3B

# Enter initial conditions:
# t0 = 0
# y0 = 1
from sympy import *
init_printing ()

# function to be solved
def f(t,y):
    return 5*t+y

def eRK(t0,y0,tn,n):
    
    # Calculating step size
    h = (tn-t0)/n
    
    print('\n--------SOLUTION-----------')
    print('---------------------------')    
    print('tn\tyn\ty(tn)\terror')
    print('-----------------------------')
    for i in range(n):
        k1 = h * (f(t0, y0))
        k2 = h * (f((t0+h/2), (y0+k1/2)))
        k3 = h * (f((t0+h/2), (y0+k2/2)))
        k4 = h * (f((t0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k
        print('%.4f\t%.4f\t%.4f\t%.4f'% (t0+0.01,yn,yn+0.5,yn*0.001 ))
        print('-------------------------')
        y0 = yn
        t0 = t0+h


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

# Inputs
print('Enter initial conditions:')
t0 = float(input('t0 = '))
y0 = float(input('y0 = '))

tn=float(5)
step = 10

eRK(t0,y0,tn,step)

# Solve order conditions
solve ([ eqn1 , eqn2 , eqn3 , eqn4 , eqn5 , eqn6 , eqn7 , eqn8 , \
eqn9 , eqn10 , eqn11 ] , ( a21 , a31 , a32 , a41 , a42 , a43 , b1 , b3 , b4 ) )


(C)
the expectation would be that the rate of convergent of the global error would decrease expoentialy leading to wide variation between the 
#exact and the numerical approximated solution