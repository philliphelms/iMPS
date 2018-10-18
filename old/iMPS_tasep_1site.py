import numpy as np
from pyscf.lib.linalg_helper import eig
from pyscf.lib.numpy_helper import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt

############################################
# Inputs
N = 2
alpha = 0.35
beta = 2./3.
s = -1.
p = 1.
maxBondDim = 50
maxIter = 2
d = 2
tol = 1e-5
plotConv = True
plotConvIn = False
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
            tmp_mat0 = einsum('ij,jk->ik',tmp_mat0,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat0[[0]]
# Diagonalize Hamiltonian
e0,lwf,rwf = la.eig(H,left=True)
inds = np.argsort(e0)
e0 = e0[inds[-1]]
rwf = rwf[:,inds[-1]]
lwf = lwf[:,inds[-1]]
#print(einsum('i,ij,j->',rwf.conj(),H,rwf)/einsum('i,i->',rwf.conj(),rwf))
#print(einsum('i,ij,j->',lwf.conj(),H,rwf)/einsum('i,i->',lwf.conj(),rwf))
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf = rwf/np.sum(rwf)
lwf = lwf/np.sum(lwf*rwf)
print('\nExact Diagonalization Energy: {}'.format(e0))
print('Energy Check {}'.format(einsum('i,ij,j->',lwf.conj(),H,rwf)/einsum('i,i->',lwf.conj(),rwf)))
############################################

############################################
# Reshape wavefunction for SVD
rpsi = np.reshape(rwf,(2,2))
lpsi = np.reshape(lwf,(2,2))
print('After Reshaping, Energy = {}'.format(einsum('ij,klim,lnjo,mo->',rpsi.conj(),W[0],W[1],rpsi)/
                                            einsum('ij,ij->',rpsi.conj(),rpsi)))
############################################

############################################
# Do SVD of initial unit cell
Ur,Sr,Vr = np.linalg.svd(rpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A = np.reshape(Ur,(a[0],d,a[1]))
A = np.swapaxes(A,0,1)
B = np.reshape(Vr,(a[1],d,a[0]))
B = np.swapaxes(B,0,1)
print('After SVD, Energy = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A.conj(),Sr,B.conj(),W[0],W[1],A,Sr,B)/
                                      einsum('jik,k,lkm,jno,o,lop->',A.conj(),Sr,B.conj(),A,Sr,B)))
# Store left and right environments
LBlock = einsum('jik,jno->ko',A.conj(),A)
RBlock = einsum('lkm,lop->ko',B.conj(),B)
LHBlock= einsum('jik,nojr,rqs->kos',A.conj(),W[0],A)
RHBlock= einsum('lkm,oplt,tsu->kos',B.conj(),W[1],B)
E = einsum('ijk,i,k,ijk->',LHBlock,Sr,Sr,RHBlock) / einsum('ko,k,o,ko->',LBlock,Sr,Sr,RBlock)
print('Energy = {}'.format(E))
############################################

############################################
converged = False
iterCnt = 0
nBond = 1
E_prev = 0
if plotConv:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
Evec = []
nBondVec = []
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
        # Push Gauge to left site --------------------------------------------
        M_reshape = np.swapaxes(wfn_rs,0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        wfn_rs = np.swapaxes(M_reshape,0,1)
        wfn_ls = einsum('klj,ji,i->kli',wfn_ls,U,s)
        # Calculate Inner F Block
        tmp_sum1 = einsum('cdf,eaf->acde',RHBlock,wfn_rs)
        tmp_sum2 = einsum('ydbe,acde->abcy',W[2],tmp_sum1)
        F = einsum('bxc,abcy->xya',np.conj(wfn_rs),tmp_sum2)
        # Create Function to give Hx
        def opt_fun(x):
            x_reshape = np.reshape(x,wfn_ls.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',F,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',LHBlock,in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0):
            return dx
        # Solve Eigenvalue Problem w/ Davidson Algorithm
        init_guess = np.reshape(wfn_ls,-1)
        u,v = eig(opt_fun,init_guess,precond,tol=tol)
        E = u/nBond
        print('\tEnergy at Left Site = {}'.format(E))
        wfn_ls = np.reshape(v,wfn_ls.shape)
        # Push Gauge to right site------------------------------------------------
        (n1,n2,n3) = wfn_ls.shape
        M_reshape = np.reshape(wfn_ls,(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        wfn_ls = np.reshape(U,(n1,n2,n3))
        wfn_rs = einsum('i,ij,kjl->kil',s,V,wfn_rs)
        # Calculate Inner f Block
        tmp_sum1 = einsum('jlp,ijk->iklp',LHBlock,np.conj(wfn_ls))
        tmp_sum2 = einsum('lmin,iklp->kmnp',W[2],tmp_sum1)
        F = einsum('npq,kmnp->kmq',wfn_ls,tmp_sum2)
        # Create Function to give Hx
        def opt_fun(x):
            x_reshape = np.reshape(x,wfn_rs.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',RHBlock,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',F,in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0):
            return dx
        # Solve Eigenvalue Problem w/ Davidson Algorithm
        init_guess = np.reshape(wfn_rs,-1)
        u,v = eig(opt_fun,init_guess,precond,tol=tol)
        E = u/nBond
        print('\tEnergy at Right Site = {}'.format(E))
        wfn_rs = np.reshape(v,wfn_rs.shape)
        if np.abs(E - Eprev_in) < tol:
            inner_converged = True
        else:
            Eprev_in = E
            if plotConvIn:
                Evec.append(E)
                nBondVec.append(nBond)
                ax1.cla()
                ax1.plot(nBondVec,Evec,'r.')
                ax2.cla()
                ax2.semilogy(nBondVec[:-1],np.abs(Evec[:-1]-Evec[-1]),'r.')
                plt.pause(0.01)

    # -----------------------------------------------------------------------------
    # Push Gauge to left site
    M_reshape = np.swapaxes(wfn_rs,0,1)
    (n1,n2,n3) = M_reshape.shape
    M_reshape = np.reshape(M_reshape,(n1,n2*n3))
    (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(n1,n2,n3))
    B = np.swapaxes(M_reshape,0,1)
    A = einsum('klj,ji->kli',wfn_ls,U)

    # -----------------------------------------------------------------------------
    # Store left and right environments
    LBlock = einsum('ij,kil,kim->lm',LBlock,A.conj(),A)
    RBlock = einsum('ijk,ilm,km->jl',B.conj(),B,RBlock)
    LHBlock= einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[2],A)
    RHBlock= einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[2],B,RHBlock)

    # ------------------------------------------------------------------------------
    # Check for convergence
    if np.abs(E - E_prev) < tol:
        converged = True
    else:
        E_prev = E
        if plotConv:
            Evec.append(E)
            nBondVec.append(nBond)
            ax1.cla()
            ax1.plot(nBondVec,Evec,'r.')
            ax2.cla()
            ax2.semilogy(nBondVec[:-1],np.abs(Evec[:-1]-Evec[-1]),'r.')
            plt.pause(0.01)
