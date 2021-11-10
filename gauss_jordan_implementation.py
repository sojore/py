# -*- coding: utf-8 -*-
"""Gauss_jordan_implementation.ipynb



-In this project we will be implementing the gauss jordan method and testing its perfomance on a sample test data

-first import the numpy library to help in the implementation, 

-we code a function for obtaining tha maximum absolute value index in an array

-we write a function for swapping rows in a matrix--we will call in this function in the implementation of the inverse function for gauss jordan

-we write a function for obtaining inverse(gauus jordan method)
-lastly we test the perfomance of the 3 functions using the given test data
"""

import numpy as np

# Function to obtain the index of the row with the maximum absolute
# value (in the column col
def index_of_max_abs(A, row=None, col=None):
  
  val=np.where(np.abs(A) == np.amax(np.abs(A))) #this function obtains both the maximum absolute value row index and the value itself
  #but we gonna only get val[0] meaning the index only 
  idx_max=val[0]

  return idx_max

# Function to swap the rows row1 and row2 in the matrix A. Since the
# matrix is passed by reference the it will be changed after calling
# the function, i.e.  no need to return a new matrix
def swap_rows(A,row1,row2):
  temp = A[row2].copy()  #we set the arrays 2rd row val to a temporary val,equate it 1st val,lastly reset it to temp,swapping the entries
  A[row2] = A[row1].copy()
  A[row1] = temp.copy()

# Implementation of the Gauss-Jordan elimination algorithm for inverse
# calculation
def inverse(A):
    # Make sure the matrix given as an argument is a square matrix
    m = A.shape[0]    # Number of rows in matrix A
    if A.shape[1] != m:
        raise Execution("Matrix must be square")

    # Create empty tildeA matrix adding the identity matrix to A
    tildeA = np.append(A, np.eye(m, dtype=np.float32), axis=1)

    # Number of columns of matrix tildeA (it should be 2*n)
    n = tildeA.shape[1]

    # # we are going to write Gaussian Elimination proc to generate an upper triangular matrix
    j = 0
    for i in range(m-1):
        pivot_element = tildeA[i][j] #remember our array this time is tildeA with rows in i and j
        if pivot_element == 0:
        
            found = False
            for k in range(i+1,m):
                if tildeA[k][j] != 0:
                    row1=i
                    row2=k
                    A=tildeA #redefining the entries so we can call the swap fuction defined earlier to swap our rows in i and k
                    swap_rows(A,row1,row2)
                    found = True 
                    break
            if found == False:
                raise Exception("The matrix is singular and hence cannot be inverted")
            else:
                pivot_element = tildeA[i][j]
        for k in range(i+1, m):
            target = tildeA[k][j]
            multiplier_val = target / pivot_element #we have out pivot element ,so we reset the target entries wrt to the pivot element
            tildeA[k] = tildeA[k] - multiplier_val*tildeA[i]
        j += 1  #incrementing the for loop

    #this code is going to generate 0s above the pivot and create a diagonal matrix
    j = m-1
    for i in range(m-1,0,-1):
        pivot_element = tildeA[i][j]
        for k in range(i-1,-1,-1): #creating a diagonal matrix whose above entries is going to be zero entries
            target = tildeA[k][j]
            multiplier_val = target / pivot_element
            tildeA[k] = tildeA[k] - multiplier_val*tildeA[i]
        j -= 1

    for i in range(m):
        tildeA[i] /= tildeA[i][i]


    # this code is going to do the extraction of the inverse matrix from the matrix tildeA
    InvA = tildeA[:,m:]

    # lastly we return the inverse matrix as stated in the reference paper
    return InvA

#testing the perfomance of the above functions using the sample test data given as per the paper

def test():
    # Test for the inverse matrix calculation
    # test = 0: sample 3x3 matrix
    # test = 1: sample 4x4 matrix
    # test = 2: sample 5x5 matrix
    # test = 3: random nxn matrix
    test =   2  # running the inverse function on test equal to 2,this value can be changed depending on the target sample test-data in question
    n = 6
    if test == 0:
      A = np.array([[0, 2, 0], [1, 4, 2], [4, 2, 0]], dtype=np.float32)
    elif test == 1:
      A = np.array([[0, 2, 0, 8], [1, 4, 2, -2], [4, 2, 0, -3], [2, -1, -6, 5]], dtype=np.float32)
    elif test == 2:
      A = np.array([[0, 2, 0, 8, 1], [1, 4, 2, -2, 5], [4, 2, 0, -3, 2], [2, -1, -6, 5, 0], [-5, 3, 8, 4, 4]], dtype=np.float32)
    else:
      A = np.random.rand(n,n)

    # Calculate the inverse matrix of A
    InvA = inverse(A)
    print('The inverse of matrix A:')
    print(A)
    print('Is:')
    print(InvA)

    print('The (rounded) product of Inv(A)*A is:')
    print(np.rint(np.matmul(A, InvA)))    # rint() rounds the result to the nearest integer
    print('The product of Inv(A)*A is:')
    print(np.matmul(A, InvA))    # rint() rounds the result to the nearest integer

    #testing the functionality of 'the index_of_max_abs '  function on the test data
    print(f'This Matrix A has a maximum absolute value index in row (s) {index_of_max_abs(A)} ')

    


if __name__ == "__main__":
    print('Running test')
    test()

#analysis

#1st:  'index_of_max_abs(a,row,col)' function is working absolutely great on the test data,providing all the rows whose index 
#has the maximum absolute value
#####  check above results for verification

#2rd : the swap_rows(A,row1,row2) function is performing as expected when called on the inverse function to perfom the swappping
# functionality of the matrix entries
#####  check above  results for verification

#3rd : the inverse(A) function is also performing pretty good on the test dataset as expected
#####  check  above results for verification

"""END OF GAUSS JORDAN IMPLEMENTATION AND TESTING.    THANK YOU!!!"""

