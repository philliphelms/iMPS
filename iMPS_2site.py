import numpy as np
from pyscf.lib.linalg_helper import eig
from pyscf.lib.numpy_helper import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt

############################################
# Inputs
alpha = 0.35
beta = 2./3.
gamma = 0.
delta = 0. 
p = 1.
q = 0.
s = -1.
maxBondDim = 10
maxIter = 2
d = 2
tol = 1e-8
plotConv = True
plotConvIn = False
hamType = 'tasep'
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
if hamType == 'tasep':
    W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
    W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
    W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
elif hamType == 'sep':
    exp_alpha = np.exp(-s)*alpha
    exp_beta = np.exp(-s)*beta
    exp_p = np.exp(-s)*p
    exp_q = np.exp(s)*q
    exp_delta = np.exp(s)*delta
    exp_gamma = np.exp(s)*gamma
    W.insert(len(W),np.array([[exp_alpha*Sm-alpha*v+exp_gamma*Sp-gamma*n, Sp, -n, Sm,-v, I]]))
    W.insert(len(W),np.array([[I                                      ],
                              [exp_p*Sm                               ],
                              [p*v                                    ],
                              [exp_q*Sp                               ],
                              [q*n                                    ],
                              [exp_delta*Sm-delta*v+exp_beta*Sp-beta*n]]))
    W.insert(len(W),np.array([[I,        z,   z, z,  z, z],
                              [exp_p*Sm, z,   z, z,  z, z],
                              [p*v,      z,   z, z,  z, z],
                              [exp_q*Sp, z,   z, z,  z, z],
                              [q*n,      z,   z, z,  z, z],
                              [z,        Sp, -n, Sm,-v, I]]))
############################################

############################################
# Make Initial Unit Cell
H = np.zeros((2**2,2**2))
occ = np.zeros((2**2,2),dtype=int)
sum_occ = np.zeros(2**2,dtype=int)
for i in range(2**2):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(2-len(bin(i)[2:]))+bin(i)[2:])))
    #print(occ[i,:])
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian
for i in range(2**2):
    i_occ = occ[i,:]
    for j in range(2**2):
        j_occ = occ[j,:]
        tmp_mat0 = np.array([[1]])
        for k in range(2):
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
U,S,V = np.linalg.svd(rpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A = np.reshape(U,(a[0],d,a[1]))
A = np.swapaxes(A,0,1)
B = np.reshape(V,(a[1],d,a[0]))
B = np.swapaxes(B,0,1)
print('After SVD, Energy = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A.conj(),S,B.conj(),W[0],W[1],A,S,B)/
                                      einsum('jik,k,lkm,jno,o,lop->',A.conj(),S,B.conj(),A,S,B)))
# Store left and right environments
LBlock = einsum('jik,jno->ko',A.conj(),A)
RBlock = einsum('lkm,lop->ko',B.conj(),B)
LHBlock= einsum('jik,nojr,rqs->kos',A.conj(),W[0],A)
RHBlock= einsum('lkm,oplt,tsu->kos',B.conj(),W[1],B)
E = einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock) / einsum('ko,k,o,ko->',LBlock,S,S,RBlock)
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
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # -----------------------------------------------------------------------------
    # Determine Hamiltonian
    H = einsum('ijk,jlmn,lopq,ros->mpirnqks',LHBlock,W[2],W[2],RHBlock)
    (n1,n2,n3,n4,n5,n6,n7,n8) = H.shape
    H = np.reshape(H,(n1*n2*n3*n4,n5*n6*n7*n8))
    # -----------------------------------------------------------------------------
    # Solve Eigenproblem
    u,v = la.eig(H)
    ind = np.argsort(u)[-1]
    E = u[ind]/nBond
    v = v[:,ind]
    print('\tEnergy at Left Site = {}'.format(E))
    # ------------------------------------------------------------------------------
    # Reshape result into state
    psi = np.reshape(v,(n1,n2,n3,n4)) # s_l s_(l+1) a_(l-1) a_(l+1)
    psi = np.swapaxes(psi,0,2) # a_(l-1) s_(l+1) s_l a_l+1
    psi = np.swapaxes(psi,1,2) # a_(l-1) s_l s_(l+1) a_(l+1)
    psi = np.swapaxes(psi,2,3) # a_(l-1) s_l a_(l+1) s_(l+1)
    psi = np.reshape(psi,(n3*n1,n4*n2))
    # ------------------------------------------------------------------------------
    # Canonicalize state
    U,S,V = np.linalg.svd(psi)
    #A = np.reshape(U,(a[0],d,-1))
    #A = A[:,:,:a[1]]
    #A = np.swapaxes(A,0,1)
    #B = np.reshape(V.T,(-1,d,a[0]))
    #B = B[
    A = np.reshape(U,(a[0],d,-1))
    A = A[:,:,:a[1]]
    A = np.swapaxes(A,0,1)
    B = np.reshape(V,(-1,d,a[0]))
    B = B[:a[-1],:,:]
    B = np.swapaxes(B,0,1)
    # -----------------------------------------------------------------------------
    # Store left and right environments
    LBlock = einsum('ij,kil,kim->lm',LBlock,A.conj(),A)
    RBlock = einsum('ijk,ilm,km->jl',B.conj(),B,RBlock)
    print(LHBlock.shape)
    print(A.conj().shape)
    print(W[2].shape)
    print(A.shape)
    LHBlock= einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[2],A)
    RHBlock= einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[2],B,RHBlock)
    #print(LHBlock.shape)
    #print(S.shape)
    #print(RHBlock.shape)
    num = einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock)
    den = einsum('ko,k,o,ko->',LBlock,S,S,RBlock)
    E = einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock) / einsum('ko,k,o,ko->',LBlock,S,S,RBlock)
    print(E)

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




# Increase Bond Dimension
#a[0] = a[1]
#a[1] = min(maxBondDim,a[0]*2)
# Put previous result into initial guess
#n1,n2,n3 = A.shape
#wfn_ls = np.pad(A,((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
#wfn_rs = np.pad(B,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
# Combine left and right side to get initial guess
#init_guess = einsum('tlr,TLR->ltTR',wfn_ls,wfn_rs)
#(n1,n2,n3,n4) = init_guess.shape
#def opt_fun(x):
#    x_reshape = np.reshape(x,(n1,n2,n3,n4))
#    in_sum1 = einsum('ijk,ilmn->jklmn',LHBlock,x_reshape)
#    in_sum2 = einsum('jopl,jklmn->kmnop',W[2],in_sum1) # could be jolp, ojpl, ojlp
#    in_sum3 = einsum('oqrm,kmnop->knpqr',W[2],in_sum2) # could be oqmr, qorm, qomr
#    fin_sum = -einsum('nqs,knpqr->kprs',RHBlock,in_sum3)#could be a lot of things...
#    return np.reshape(fin_sum,-1)
#def precond(dx,e,x0):
#    return dx
#init_guess = np.reshape(init_guess,-1)
#u,v = eig(opt_fun,init_guess,precond,tol=tol)
#E = u/nBond
#print('\tEnergy at Left Site = {}'.format(E))
#psi = np.reshape(v,(n1,n2,n3,n4))
#psi = np.swapaxes(psi,2,3)
#psi = np.reshape(psi,(d*a[0],d*a[0]))
#print(psi.shape)

#print('After SVD, Energy = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A.conj(),Sr,B.conj(),W[0],W[1],A,Sr,B)/
#                                  einsum('jik,k,lkm,jno,o,lop->',A.conj(),Sr,B.conj(),A,Sr,B)))

