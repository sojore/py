# -*- coding: utf-8 -*-
"""Gram-Schmidt Algorithm and QR decomposition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WE176ZIYJ_f4QHA8HpsqJCHGT3bo8ZFp
"""

import numpy as np

#function to generate random matrix A
A=np.random.random((3,4))

#function for calculating the frobenious norm of matrix A
norm_val = np.linalg.norm(A)

print(f'The calculated norm value of matrix A {A} is = {norm_val} and with a  dimension(s) of {A.ndim}')

#implementing the Gram-Schmidt Algorithm 
#this function will be printing  appropriate messages for Gram-Schmidt algorithm applicability on columns of the matrix A
#and then we use the same function to generate the matrix Q from A
import numpy as np
def gram_schmidt_algorithm(A):
  matrix_Q = []
  for i in range(len(A)):
    
    #this perfoms the normalization approach
    matrix_Q_orth = A[i]
    for j in range(len(matrix_Q)):
      matrix_Q_orth = matrix_Q_orth - (matrix_Q[j] @ A[i])*matrix_Q[j]
      if np.sqrt(sum(matrix_Q_orth**2)) <= 1e-10: #here we will be checking for linearly independence 
        print('The given vector is linearly dependent.')
        return matrix_Q
    # performing the Gram-Schmidt orthogonalization
    else:
      matrix_Q_orth = matrix_Q_orth / np.sqrt(sum(matrix_Q_orth**2))
      matrix_Q.append(matrix_Q_orth)

  print('The given vector is linearly independent.')
  return matrix_Q

Q = gram_schmidt_algorithm(A)

# printing  appropriate messages for Gram-Schmidt algorithm applicability on columns of the matrix Q
if (sum(Q[0]**2))**0.5<=0:
  print(f'The Gram-Schmidt algorithm is not applicable on the matrix A from the first columns of A')
else:
  print(f'The Gram-Schmidt algorithm  is applicable from the first column of A')

if Q[0] @ Q[1]<=0:
  print(f'The Gram-Schmidt algorithm is not applicable on the matrix A from the inner columns of A')
else:
  print(f'The Gram-Schmidt algorithm  is applicable from the inner columns of A')

if (sum(Q[2]**2))**0.5<=0:
  print(f'The Gram-Schmidt algorithm is not applicable on the matrix A from the last columns of A')
else:
  print(f'The Gram-Schmidt algorithm  is applicable from the last columns of A')

#printing the matrix Q from A
print(f'The matrix Q produced by gram_schmidt_algorithm from A is{Q} ')

def QR_decomposition(A):
  Matrix_Q_transpose = np.array(gram_schmidt_algorithm(A.T))
  tranposed_matrix = Matrix_Q_transpose @ A
  Matrix_Q = Matrix_Q_transpose.T
  return Matrix_Q, tranposed_matrix

Matrix_Q, tranposed_matrix = QR_decomposition(A)
print(f'From QR_decomposition the matrix Q is {Matrix_Q} and matrix R is {tranposed_matrix} ')
