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
maxIter = 2
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
#print(np.einsum('i,ij,j->',rwf.conj(),H,rwf)/np.einsum('i,i->',rwf.conj(),rwf))
#print(np.einsum('i,ij,j->',lwf.conj(),H,rwf)/np.einsum('i,i->',lwf.conj(),rwf))
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf = rwf/np.sum(rwf)
lwf = lwf/np.sum(lwf*rwf)
print('\nExact Diagonalization Energy: {}'.format(e0))
print('Energy Check {}'.format(np.einsum('i,ij,j->',lwf.conj(),H,rwf)/np.einsum('i,i->',lwf.conj(),rwf)))
############################################

############################################
# Reshape wavefunction for SVD
rpsi = np.reshape(rwf,(2,2))
lpsi = np.reshape(lwf,(2,2))
print('After Reshaping, Energy = {}'.format(np.einsum('ij,klim,lnjo,mo->',rpsi.conj(),W[0],W[1],rpsi)/
                                            np.einsum('ij,ij->',rpsi.conj(),rpsi)))
############################################

############################################
# Do SVD of initial unit cell
Ur,Sr,Vr = np.linalg.svd(rpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A = np.reshape(Ur,(a[0],d,a[1]))
B = np.reshape(Vr,(a[1],d,a[0]))
print('After SVD, Energy = {}'.format(np.einsum('ijk,k,klm,nojr,oplt,qrs,s,stu->',A.conj(),Sr,B.conj(),W[0],W[1],A,Sr,B)/
                                      np.einsum('ijk,k,klm,njo,o,olp->',A.conj(),Sr,B.conj(),A,Sr,B)))
# Store left and right environments
LBlock = np.einsum('ijk,njo->ko',A.conj(),A)
RBlock = np.einsum('klm,olp->ko',B.conj(),B)
LHBlock= np.einsum('ijk,nojr,qrs->kos',A.conj(),W[0],A)
RHBlock= np.einsum('klm,oplt,stu->kos',B.conj(),W[1],B)
E = np.einsum('ijk,i,k,ijk->',LHBlock,Sr,Sr,RHBlock) / np.einsum('ko,k,o,ko->',LBlock,Sr,Sr,RBlock)
print('Energy = {}'.format(E))
############################################

############################################
converged = False
iterCnt = 0
while not converged:
    # Minimize energy wrt Hamiltonian
    def dot_twosite(LHBlock, RHBlock, W1, W2, x):
        #   O-i-O---O-n-O
        #   |   |   |   |
        #   |   l   m   |
        #   |   |   |   |
        #   O-j-O-o-O-q-O
        #   |   |   |   |
        #   |   p   r   |
        #   |           |
        #   O-k       s-O
        print(x)
        tmpsum1 = np.einsum("ijk, ilmn-> jklmn", LHBlock, x)
        tmpsum2 = np.einsum("jklmn, jopl -> kmnop", tmpsum1, W1)
        tmpsum3 = np.einsum("kmnop, oqrm -> knpqr", tmpsum2, W2)
        return np.einsum("knpqr, nqs -> kprs", tmpsum3, RHBlock)
    # Increase Bond Dimension
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # Put previous result into initial guess
    n1,n2,n3 = A.shape
    wfn_ls = np.pad(A,((0,a[0]-n1),(0,0),(0,a[1]-n3)),'constant')
    print(wfn_ls.shape)
    wfn_rs = np.pad(B,((0,a[1]-n3),(0,0),(0,a[0]-n1)),'constant')
    print(wfn_rs.shape)
    wfn2 = np.einsum('ijk,klm->ijlm',wfn_ls,wfn_rs)
    print(wfn2)
    mps_shape = wfn2.shape
    def dot_flat(x):
        return np.reshape(dot_twosite(LHBlock, RHBlock, W[2], W[2], x.reshape(mps_shape)),-1)
    def precond(dx,e,x0):
        return dx
    energy, wfn0 = eig(dot_flat, wfn2.ravel(), precond)
    print(energy)
    wfn0 = wfn0.reshape((a[-1]*d,a[-1]*d))
    # Do SVD of the unit cell
    a.append(min(maxBondDim,a[-1]*d))
    U,S,V = np.linalg.svd(wfn0,full_matrices=True)
    U = np.reshape(U,(a[-2],d,-1))
    V = np.reshape(V,(-1,d,a[-2]))
    U = U[:,:,:a[-1]]
    S = S[:a[-1]]
    V = V[:a[-1],:,:]
    A = U
    B = V
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
