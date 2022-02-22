import numpy as np 
import matplotlib.pyplot as plt
n = 50
p = 0.45
q = 0.05

# generate stochastic block model matrix
iu1 = np.triu_indices(n,1)
W_AA = np.zeros((n,n))
W_BA = np.zeros((n,n))
W_BB = np.zeros((n,n))
#W_CC = np.zeros((n,n))

# set random binary edges for upper triangle of matrix
W_AA[iu1]  = np.random.binomial(1,p,int(n*(n-1)/2))
W_BB[iu1]  = np.random.binomial(1,p,int(n*(n-1)/2))
W_BA = np.random.binomial(1,q,size = (n,n))
W = np.concatenate((np.concatenate((W_AA,W_BA.T),axis = 1),np.concatenate((W_BA,W_BB),axis = 1)), axis = 0)

# Graph is unweighted - make matrix symmetric
W = W + W.transpose()
L = np.diag(np.sum(W,axis = 1))-W

# compute eigenvalues and eigenvectors of Laplacian
[D,V] = np.linalg.eigh(L)

#show matrix, Fiedler vector and eigenvalues
fig,ax = plt.subplots(1,3,figsize = (15,5))
ax[0].imshow(W)
ax[1].plot(V[:,1])
ax[2].plot(np.abs(D[0:10]),'bo')
plt.show()
a = 1




