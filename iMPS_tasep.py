import numpy as np
from pyscf.lib.linalg_helper import eig
from scipy import linalg as la

############################################
# Inputs
N = 2
alpha = 0.35
beta = 2./3.
s = -1.
p = 1.
maxBondDim = 10
maxIter = 100
d = 2
############################################

############################################
# Determine MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
############################################

############################################
# Make Initial Unit Cell
H = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N,dtype=int)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    #print(occ[i,:])
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat0 = np.array([[1]])
        for k in range(N):
            tmp_mat0 = np.einsum('ij,jk->ik',tmp_mat0,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat0[[0]]
# Diagonalize Hamiltonian
e0,lwf,rwf = la.eig(H,left=True)
inds = np.argsort(e0)
e0 = e0[inds[-1]]
rwf = rwf[:,inds[-1]]
lwf = lwf[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf = rwf/np.sum(rwf)
lwf = lwf/np.sum(lwf*rwf)
print('\nExact Diagonalization Energy: {}'.format(e0))
############################################

############################################
# Reshape wavefunction for SVD
psi = np.reshape(rwf,(2,2))
############################################

############################################
# Do SVD of initial unit cell
U,S,V = np.linalg.svd(psi)
A = []
B = []
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A.append(np.reshape(U,(a[0],d,a[1])))
B.append(np.reshape(V,(a[1],d,a[0])))
# Store left and right environments
LBlock = np.einsum('ijk,ljm->km',A[0],A[0].conj())
RBlock = np.einsum('ijk,ljm->il',B[0],B[0].conj())
LnormBlock = np.einsum('ijk->k',A[0])
RnormBlock = np.einsum('ijk->i',B[0])
LHBlock = np.einsum('ijk,lmjn,onp->kmp',A[0],W[0],A[0].conj())
RHBlock = np.einsum('ijk,lmjn,onp->ilo',B[0],W[1],B[0].conj())
# Calculate Normalized Energy?
E = np.einsum('kmp,k,p,kmp->',LHBlock,S,S,RHBlock)
norm = np.einsum('i,i,i->',LnormBlock,S,RnormBlock)
print('Energy = {}'.format(E))
print('Normalization Factor = {}'.format(norm))
print('E/norm = {}'.format(E/norm))
############################################

############################################
converged = False
iterCnt = 0
while not converged:
    # Minimize energy wrt Hamiltonian
    def dot_twosite(LHBlock, RHBlock, W1, W2, x):
        tmpsum1 = np.einsum("ijk, ilmn-> jklmn", LHBlock, x)
        tmpsum2 = np.einsum("jklmn, jopl -> kmnop", tmpsum1, W1)
        tmpsum3 = np.einsum("kmnop, oqrm -> knpqr", tmpsum2, W2)
        return np.einsum("knpqr, nqs -> kprs", tmpsum3, RHBlock)
    a[0] = a[1]
    a[1] = min(maxBondDim,2**(iterCnt+1))
    wfn2 = np.random.rand(a[1],d,d,a[-1])
    mps_shape = wfn2.shape
    def dot_flat(x):
        return dot_twosite(LHBlock, RHBlock, W[1], W[1], x.reshape(mps_shape)).ravel()
    def precond(dx,e,x0):
        return dx
    energy, wfn0 = eig(dot_flat, wfn2.ravel(), precond)
    wfn0 = wfn0.reshape((a[-1]*d,a[-1]*d))
    # Do SVD of the unit cell
    a.append(min(maxBondDim,a[-1]*d))
    U,S,V = np.linalg.svd(wfn0,full_matrices=True)
    U = np.reshape(U,(a[-2],d,-1))
    V = np.reshape(V,(-1,d,a[-2]))
    U = U[:,:,:a[-1]]
    S = S[:a[-1]]
    V = V[:a[-1],:,:]
    A.append(U)
    B.append(V)
    LBlock = np.einsum('il,ijk,ljm->km',LBlock,A[-1],A[-1].conj())
    RBlock = np.einsum('ijk,ljm,km->il',B[-1],B[-1].conj(),RBlock)
    LnormBlock = np.einsum('i,ijk->k',LnormBlock,A[-1])
    RnormBlock = np.einsum('ijk,k->i',B[-1],RnormBlock)
    LHBlock = np.einsum('ilo,ijk,lmjn,onp->kmp',LHBlock,A[-1],W[1],A[-1].conj())
    RHBlock = np.einsum('ijk,lmjn,onp,kmp->ilo',B[-1],W[1],B[-1].conj(),RHBlock)
    iterCnt += 1
    if iterCnt > maxIter:
        converged = True
    # Calculate Normalized Energy?
    E = np.einsum('kmp,k,p,kmp->',LHBlock,S,S,RHBlock)
    norm = np.einsum('i,i,i->',LnormBlock,S,RnormBlock)
    print('Energy = {}'.format(E))
    print('Normalization Factor = {}'.format(norm))
    print('E/norm = {}'.format(E/norm))
############################################

############################################
# Optimize Unit Cell
############################################

############################################
# Push Unit Cell Outwards
############################################

############################################
# Initialize Unit Cell
############################################
