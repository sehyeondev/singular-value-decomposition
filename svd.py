import numpy as np

# Exercise 11.3.1 in the MMDS textbook
# original matrix
m = np.array([[1,2,3], [3,4,5], [5,4,3], [0,2,4], [1,3,5]])
print ("original m")
print (m.__repr__())

# (a) comput mTm and mmT
mTm = np.dot(np.transpose(m), m)
mmT = np.dot(m, np.transpose(m))
print("(a) compute mTm and mmT")
print (mTm.__repr__())
print (mmT.__repr__())

# (b) find eigenpairs for mTm and mmT
w1, v1 = np.linalg.eig(mTm) # to find V
w2, v2 = np.linalg.eig(mmT) # to find U
print ("(b) eigenpairs for mTm and mmT")
print("eigpairs for mTm")
print(w1)
print(v1)
print("eigenpairs for mmT")
print(w2)
print(v2)

# (c) find SVD for original matrix
rank = np.linalg.matrix_rank(m)
print("rank")
print(rank)

V = np.zeros((m.shape[1], rank)) # r by n matrix
sigma_sq = np.zeros((rank, rank)) # r by r matrix
U = np.zeros((m.shape[0], rank)) # m by r matrix
eig_values = []

w1_index = [] # indexes of largest eigenvalues as many as rank from mTm
w2_index = [] # indexes of largest eigenvalues as many as rank from mmT
for i in range(rank):
    index_w1 = list(w1).index(max(w1))
    w1_index.append(index_w1)
    eig_values.append(w1[w1_index[i]])
    w1[index_w1] = 0
    index_w2 = list(w2).index(max(w2))
    w2_index.append(index_w2)
    w2[index_w2] = 0

for i in range(rank):
    V[:,i] = v1[:,w1_index[i]]
    U[:,i] = v2[:,w2_index[i]]    
if V[0,0] < 0: V = -V
if U[0,0] < 0: U = -U
sigma_sq = np.diag(eig_values)
sigma = np.sqrt(sigma_sq)
tot_energy = sum(eig_values)

print("(c) decomposed V, sigma, U")
print(V.__repr__())
print(sigma.__repr__())
print(U.__repr__())

# (d) set smaller singular value as 0
#     and compute the one-dimensional approximation to original matrix
sigma[1][1] = 0
approx_m = np.dot(np.dot(U, sigma), np.transpose(V))
retained_energy = np.square(sigma[0][0])
print ("(d) updated sigma and M")
print(sigma.__repr__())
print (approx_m.__repr__())

# (e) compute energy ratio of singular values of approxmation to that of original
ratio_energy = retained_energy/tot_energy
print ("(e) compute total E, retained E and ratio")
print (tot_energy)
print (retained_energy)
print (ratio_energy)