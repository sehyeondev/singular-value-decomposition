import numpy as np

# Exercise 11.3.1 in the MMDS textbook
# original matrix
m = np.array([[1,2,3], [3,4,5], [5,4,3], [0,2,4], [1,3,5]])
# print ("original m")
# print (m.__repr__())

# (a) comput mTm and mmT
mTm = np.dot(np.transpose(m), m)
mmT = np.dot(m, np.transpose(m))
# print("compare mTm and mmT")
# print (mTm.__repr__())
# print (mmT.__repr__())

# (b) find eigenpairs for mTm and mmT
w1, v1 = np.linalg.eig(mTm)
w2, v2 = np.linalg.eig(mmT)
# print("eigpairs for mTm")
# print(w1)
# print(v1)
# print("eigenpairs for mmT")
# print(w2)
# print(v2)

# (c) find SVD for original matrix
V = v1
sigma_sq = np.diag(w1)
sigma = np.sqrt(sigma_sq)
U = v2
print("results below are V, sigma, U in order")
print(V.__repr__())
print(sigma.__repr__())
print(U.__repr__())

# (d) set smaller singular value as 0
#     and compute the one-dimensional approximation to original matrix

# (e) compute energy ratio of singular values of approxmation to that of original