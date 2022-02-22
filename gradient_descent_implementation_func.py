# -*- coding: utf-8 -*-
"""gradient_descent_implementation_func.ipynb



#in this project we will be implementing gradient descent algorithm for the given function,**see attached pdf
#so we proceed as below

#first we import some important libraries we gonna be using for implementation
import numpy as np
from numpy import asarray,arange
import matplotlib.pyplot as plt
from numpy.random import rand

#gradient descent implementation of one-dimensional function 
#defining the problem function **see attached pdf
def function_problem_def(x):
	return (-20*(np.exp(1)**(-0.125*(x**2))))-(np.exp(1)**(0.5*np.cos(2*np.pi*x)))+20+np.exp(1)
 
# calculating the derivertive of the given function ***see attached pdf for its derivatition
def function_derivertive(x):
	return (5*x*(np.exp(1)**(-0.125*(x**2))))
 
# defing the gradient descent algorithm itself
def gradient_descent_algorithm(function_problem_def, function_derivertive, stopping_val, number_of_iter, step_size):
	points, score_val = list(), list() ##this will be storing the values of the outcome 
	sol = stopping_val[:, 0] + rand(len(stopping_val)) * (stopping_val[:, 1] - stopping_val[:, 0]) #getting the initial starting point of the algorithm 
	for i in range(number_of_iter):
		gradient_value = function_derivertive(sol)  #this calculates the gradient of the given function 

		sol = sol - step_size * gradient_value  ##taking a step size during the gradient values approximation 
		solution_eval = function_problem_def(sol)
  
    #we will be appending the obtained soltuions into our list defined earlier
		points.append(sol)
		score_val.append(solution_eval)

		print('>%d reducing minimum point(%s) = %.5f' % (i, sol, solution_eval))
  
  
	return [points, score_val]
 


#initializing the number of iterations
number_of_iter = 11

#initializing the step size for the gradient descent  1e-5
step_size = 0.1

#determing our stopping criterian to be bound on the said domain
stopping_val = asarray([[-1.0, 1.0]])

#running the gradient descent algorithm to get the minimum point
points, score_val = gradient_descent_algorithm(function_problem_def, function_derivertive, stopping_val, number_of_iter, step_size)

inputs = arange(stopping_val[0,0], stopping_val[0,1]+0.1, 0.1)#increamenting the inputs from  bound ranges by a value of 0.1
final_val = function_problem_def(inputs)

print('The above last array([****]) point represents the obtained minimum point!')
#plotting the graph for a better visualization of the local minimum point to be obtained
plt.plot(inputs, final_val,color='green')
plt.plot(points, score_val, '.-', color='red')
plt.grid()
plt.show()

#make sure to compare the average outcome to obtain the local minimum point
#as found on the function derivertive, minimum point for this function is at aound the  0.0 value mark

"""END OF GRADIENT DESCENT ALGORITHM IMPLEMETATION AND EVALUATION  ****THANK YOU!!!***"""

