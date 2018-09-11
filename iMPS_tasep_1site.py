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
tol = 1e-3
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
A = np.swapaxes(A,0,1)
B = np.reshape(Vr,(a[1],d,a[0]))
B = np.swapaxes(B,0,1)
print('After SVD, Energy = {}'.format(np.einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A.conj(),Sr,B.conj(),W[0],W[1],A,Sr,B)/
                                      np.einsum('jik,k,lkm,jno,o,lop->',A.conj(),Sr,B.conj(),A,Sr,B)))
# Store left and right environments
LBlock = np.einsum('jik,jno->ko',A.conj(),A)
RBlock = np.einsum('lkm,lop->ko',B.conj(),B)
LHBlock= np.einsum('jik,nojr,rqs->kos',A.conj(),W[0],A)
RHBlock= np.einsum('lkm,oplt,tsu->kos',B.conj(),W[1],B)
E = np.einsum('ijk,i,k,ijk->',LHBlock,Sr,Sr,RHBlock) / np.einsum('ko,k,o,ko->',LBlock,Sr,Sr,RBlock)
print('Energy = {}'.format(E))
############################################

############################################
converged = False
iterCnt = 0
nBond = 1
Eprev = 0
while not converged:
    nBond += 2
    # Minimize energy wrt Hamiltonian
    # Increase Bond Dimension
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # Put previous result into initial guess
    n1,n2,n3 = A.shape
    wfn_ls = np.pad(A,((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
    wfn_rs = np.pad(B,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
    # Jump between left and right site optimizations
    inner_converged = False
    Eprev_in = 0
    while not inner_converged:
        # Push Gauge to left site
        M_reshape = np.swapaxes(wfn_rs,0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        wfn_rs = np.swapaxes(M_reshape,0,1)
        wfn_ls = np.einsum('klj,ji,i->kli',wfn_ls,U,s)
        # Calculate Inner F Block
        tmp_sum1 = np.einsum('cdf,eaf->acde',RHBlock,wfn_rs)
        tmp_sum2 = np.einsum('ydbe,acde->abcy',W[2],tmp_sum1)
        F = np.einsum('bxc,abcy->xya',np.conj(wfn_rs),tmp_sum2)
        # Determine Hamiltonian
        H = np.einsum('jlp,lmin,kmq->ijknpq',LHBlock,W[2],F)
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        # Solve Eigenvalue Problem
        u,v = np.linalg.eig(H)
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]/nBond
        v = v[:,max_ind]
        print('\tEnergy at Left Site = {}'.format(E))
        wfn_ls = np.reshape(v,(n1,n2,n3))
        # Push Gauge to right site
        M_reshape = np.reshape(wfn_ls,(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        wfn_ls = np.reshape(U,(n1,n2,n3))
        wfn_rs = np.einsum('i,ij,kjl->kil',s,V,wfn_rs)
        # Calculate Inner f Block
        F = np.einsum('jlp,ijk,lmin,npq->kmq',LHBlock,np.conj(wfn_ls),W[2],wfn_ls)
        # Determine Hamiltonian
        H = np.einsum('jlp,lmin,kmq->ijknpq',F,W[2],RHBlock)
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,v = np.linalg.eig(H)
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]/nBond
        v = v[:,max_ind]
        print('\tEnergy at Right Site= {}'.format(E))
        wfn_rs = np.reshape(v,(n1,n2,n3))
        if np.abs(E - Eprev_in) < tol:
            inner_converged = True
        else:
            Eprev_in = E

    # Push Gauge to left site
    M_reshape = np.swapaxes(wfn_rs,0,1)
    (n1,n2,n3) = M_reshape.shape
    M_reshape = np.reshape(M_reshape,(n1,n2*n3))
    (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(n1,n2,n3))
    B = np.swapaxes(M_reshape,0,1)
    A = np.einsum('klj,ji->kli',wfn_ls,U)

    # Store left and right environments
    LBlock = np.einsum('ij,kil,kim->lm',LBlock,A.conj(),A)
    RBlock = np.einsum('ijk,ilm,km->jl',B.conj(),B,RBlock)
    LHBlock= np.einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[2],A)
    RHBlock= np.einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[2],B,RHBlock)
    E = np.einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock) / np.einsum('ko,k,o,ko->',LBlock,S,S,RBlock)/nBond
    print('Energy = {},{}'.format(E,nBond))

    if np.abs(E - Eprev) < tol:
        converged = True
    else:
        E_prev = E
    
