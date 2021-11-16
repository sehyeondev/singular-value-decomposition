import numpy as np

# Exercise 11.3.1 in the MMDS textbook
# original matrix
m = np.array([[1,2,3], [3,4,5], [5,4,3], [0,2,4], [1,3,5]])
# (a) compute mTm and mmT
mTm = np.dot(np.transpose(m), m)
mmT = np.dot(m, np.transpose(m))
# (b) find eigenpairs for mTm and mmT
w1, v1 = np.linalg.eig(mTm) # to find V
w2, v2 = np.linalg.eig(mmT) # to find U
# (c) find SVD for original matrix
rank = np.linalg.matrix_rank(m) # rank = 2
V = np.zeros((m.shape[1], rank)) # r by n matrix
sigma_sq = np.zeros((rank, rank)) # r by r matrix
U = np.zeros((m.shape[0], rank)) # m by r matrix
eig_values = []
w1_index = [] # indexes of largest eigenvalues as many as rank from mTm
w2_index = [] # indexes of largest eigenvalues as many as rank from mmT
# get largest eigenvalues as many as rank from mTm and mmT
for i in range(rank):
    index_w1 = list(w1).index(max(w1))
    w1_index.append(index_w1)
    eig_values.append(w1[w1_index[i]])
    w1[index_w1] = 0
    index_w2 = list(w2).index(max(w2))
    w2_index.append(index_w2)
    w2[index_w2] = 0
# get V, U, and sigma
for i in range(rank):
    V[:,i] = v1[:,w1_index[i]]
    U[:,i] = v2[:,w2_index[i]]    
if V[0,0] < 0: V = -V
if U[0,0] < 0: U = -U
sigma_sq = np.diag(eig_values)
sigma = np.sqrt(sigma_sq)
# (d) set smaller singular value as 0
#     and compute the one-dimensional approximation to original matrix
sigma[1][1] = 0
approx_m = np.dot(np.dot(U, sigma), np.transpose(V))
# (e) compute energy ratio of singular values of approxmation to that of original
tot_energy = sum(eig_values)
retained_energy = np.square(sigma[0][0])
ratio_energy = retained_energy/tot_energy

print("(a)-1 compute mTm")
print (mTm.__repr__())
print("(a)-2 compute mmT")
print (mmT.__repr__())
print("(b)-1 eigenvalues for mTm")
print(w1)
print("(b)-2 eigenvectors for mTm")
print(v1)
print("(b)-3 eigenvalues for mmT")
print(w2)
print("(b)-4 eigenvectors for mmT")
print(v2)
print("(c)-1 result of SVD: V")
print(V.__repr__())
print("(c)-2 result of SVD: sigma")
print(sigma.__repr__())
print("(c)-3 result of SVD: U")
print(U.__repr__())
print ("(d)-1 updated sigma")
print(sigma.__repr__())
print ("(d)-2 one dimensional approximation to M")
print (approx_m.__repr__())
print ("(e)-1 total Energy")
print (tot_energy)
print ("(e)-2 retained Energy")
print (retained_energy)
print ("(e)-3 (retained E)/(total E)")
print (ratio_energy)