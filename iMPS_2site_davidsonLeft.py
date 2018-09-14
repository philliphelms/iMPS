import numpy as np
from pyscf.lib.linalg_helper import eig
#from pyscf.lib.numpy_helper import einsum
from numpy import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt

############################################################################
# General Simple Exclusion Process:

#                     _p_
#           ___ ___ _|_ \/_ ___ ___ ___ ___ ___
# alpha--->|   |   |   |   |   |   |   |   |   |---> beta
# gamma<---|___|___|___|___|___|___|___|___|___|<--- delta
#                   /\___|
#                      q 
#
#
###########################################################################

############################################
# Inputs
alpha = 0.9
beta = 0.9
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
hamType = 'sep'
############################################

############################################
# Determine MPO
# Basic Operators
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
# Exponentially weighted hopping rates
exp_alpha = np.exp(-s)*alpha
exp_beta = np.exp(-s)*beta
exp_p = np.exp(-s)*p
exp_q = np.exp(s)*q
exp_delta = np.exp(s)*delta
exp_gamma = np.exp(s)*gamma
# MPO Lists
W = []
Wl = []
if hamType == 'tasep':
    W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
    W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
    W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
elif hamType == 'sep':
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
Wl.insert(len(Wl),np.transpose(W[0],(0,1,3,2)).conj())
Wl.insert(len(Wl),np.transpose(W[1],(0,1,3,2)).conj())
Wl.insert(len(Wl),np.transpose(W[2],(0,1,3,2)).conj())
############################################

############################################
# Current & Density Operators
currentOp = [None]*2
currentOp[0] = np.array([[Sp,Sm]])
currentOp[1] = np.array([[exp_p*Sm],
                         [exp_q*Sp]])
densityLOp = [None]*2
densityLOp[0] = np.array([[n]])
densityLOp[1] = np.array([[I]])
densityROp = [None]*2
densityROp[0] = np.array([[I]])
densityROp[1] = np.array([[n]])
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
e0,lpsi,psi = la.eig(H,left=True)
inds = np.argsort(e0)
e0 = e0[inds[-1]]
psi = psi[:,inds[-1]]
lpsi = lpsi[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
psi = psi/np.sum(psi)
lpsi = lpsi/np.sum(lpsi*psi)
print('\nExact Diagonalization Energy: {}'.format(e0))
print('Energy Check {}'.format(einsum('i,ij,j->',lpsi.conj(),H,psi)/einsum('i,i->',lpsi.conj(),psi)))
############################################

############################################
# Reshape wavefunction for SVD
psi = np.reshape(psi,(2,2))
lpsi = np.reshape(lpsi,(2,2))
print('After Reshaping, Energy = {}'.format(einsum('ij,klim,lnjo,mo->',psi.conj(),W[0],W[1],psi)/
                                            einsum('ij,ij->',psi.conj(),psi)))
############################################

############################################
# Do SVD of initial unit cell
U,S,V = np.linalg.svd(psi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A = np.reshape(U,(a[0],d,a[1]))
A = np.swapaxes(A,0,1)
B = np.reshape(V,(a[1],d,a[0]))
B = np.swapaxes(B,0,1)
print('After SVD, Energy = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A.conj(),S,B.conj(),W[0],W[1],A,S,B)/
                                      einsum('jik,k,lkm,jno,o,lop->',A.conj(),S,B.conj(),A,S,B)))
# Left
Ul,Sl,Vl = np.linalg.svd(lpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
Al = np.reshape(Ul,(a[0],d,a[1]))
Al = np.swapaxes(Al,0,1)
Bl = np.reshape(Vl,(a[1],d,a[0]))
Bl = np.swapaxes(Bl,0,1)
print('After SVD, Energy = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',Al.conj(),Sl,Bl.conj(),Wl[0],Wl[1],Al,Sl,Bl)/
                                      einsum('jik,k,lkm,jno,o,lop->',Al.conj(),Sl,Bl.conj(),Al,Sl,Bl)))
############################################

############################################
# Store left and right environments
LBlock = einsum('jik,jno->ko',A.conj(),A)
RBlock = einsum('lkm,lop->ko',B.conj(),B)
LHBlock= einsum('jik,nojr,rqs->kos',A.conj(),W[0],A)
RHBlock= einsum('lkm,oplt,tsu->kos',B.conj(),W[1],B)
E = einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock) / einsum('ko,k,o,ko->',LBlock,S,S,RBlock)
print('Energy = {}'.format(E))
# Left
LBlockl = einsum('jik,jno->ko',Al.conj(),Al)
RBlockl = einsum('lkm,lop->ko',Bl.conj(),Bl)
LHBlockl= einsum('jik,nojr,rqs->kos',Al.conj(),Wl[0],Al)
RHBlockl= einsum('lkm,oplt,tsu->kos',Bl.conj(),Wl[1],Bl)
El = einsum('ijk,i,k,ijk->',LHBlockl,Sl,Sl,RHBlockl) / einsum('ko,k,o,ko->',LBlockl,Sl,Sl,RBlockl)
print('Energy = {}'.format(El))
# Left & Right
LBlocklr = einsum('jik,jno->ko',Al.conj(),A)
RBlocklr = einsum('lkm,lop->ko',Bl.conj(),B)
LHBlocklr = einsum('jik,nojr,rqs->kos',Al.conj(),W[0],A)
RHBlocklr = einsum('lkm,oplt,tsu->kos',Bl.conj(),W[1],B)
Elr = einsum('ijk,i,k,ijk->',LHBlocklr,Sl,S,RHBlocklr) / einsum('ko,k,o,ko->',LBlocklr,Sl,S,RBlocklr)
print('Energy = {}'.format(Elr))
############################################

############################################
converged = False
iterCnt = 0
nBond = 1
E_prev = 0
El_prev = 0
if plotConv:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
Evec = []
Elvec = []
nBondVec = []
while not converged:
    # -----------------------------------------------------------------------------
    # Some Prerequisites
    nBond += 2
    iterCnt += 1
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # -----------------------------------------------------------------------------
    # Determine Initial Guess
    # Pad A and B
    (n1,n2,n3) = A.shape
    Aguess = np.pad(einsum('ijk,k->ijk',A,S),((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
    Bguess = np.pad(B,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
    initGuess = einsum('ijk,lkm->iljm',Aguess,Bguess)
    guessShape = initGuess.shape
    initGuess = initGuess.ravel()
    # Left
    Alguess = np.pad(einsum('ijk,k->ijk',Al,Sl),((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
    Blguess = np.pad(Bl,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
    initGuessl = einsum('ijk,lkm->iljm',Alguess,Blguess)
    initGuessl = initGuessl.ravel()
    # -----------------------------------------------------------------------------
    # Determine Hamiltonian Function
    def Hx(x):
        x_reshape = np.reshape(x,guessShape)
        tmp1 = einsum('ijk,nqks->ijnqs',LHBlock,x_reshape) # Could be 'ijk,mpir->jkmpr'
        tmp2 = einsum('jlmn,ijnqs->ilmqs',W[2],tmp1)
        tmp3 = einsum('lopq,ilmqs->imops',W[2],tmp2)
        finalVec = einsum('ros,imops->mpir',RHBlock,tmp3)
        return -finalVec.ravel()
    def Hlx(x):
        x_reshape = np.reshape(x,guessShape)
        tmp1 = einsum('ijk,nqks->ijnqs',LHBlockl,x_reshape) # Could be 'ijk,mpir->jkmpr'
        tmp2 = einsum('jlmn,ijnqs->ilmqs',Wl[2],tmp1)
        tmp3 = einsum('lopq,ilmqs->imops',Wl[2],tmp2)
        finalVec = einsum('ros,imops->mpir',RHBlockl,tmp3)
        return -finalVec.ravel()
    def precond(dx,e,x0):
        return dx
    # -----------------------------------------------------------------------------
    # Solve Eigenproblem
    u,v = eig(Hx,initGuess,precond) # PH - Add tolerance here?
    E = -u/nBond
    print('\tEnergy from Optimization = {}'.format(E))
    # Left
    ul,vl = eig(Hlx,initGuessl,precond) # PH - Add tolerance here?
    El = -ul/nBond
    print('\tEnergy from Optimization = {}'.format(El))
    # ------------------------------------------------------------------------------
    # Reshape result into state
    psi = np.reshape(v,(d,d,a[0],a[0])) # s_l s_(l+1) a_(l-1) a_(l+1)
    psi = np.transpose(psi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
    psi = np.reshape(psi,(a[0]*d,a[0]*d))
    # Left
    lpsi = np.reshape(vl,(d,d,a[0],a[0])) # s_l s_(l+1) a_(l-1) a_(l+1)
    lpsi = np.transpose(lpsi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
    lpsi = np.reshape(lpsi,(a[0]*d,a[0]*d))
    # ------------------------------------------------------------------------------
    # Canonicalize state
    U,S,V = np.linalg.svd(psi)
    A = np.reshape(U,(a[0],d,-1))
    A = A[:,:,:a[1]]
    A = np.swapaxes(A,0,1)
    B = np.reshape(V,(-1,d,a[0]))
    B = B[:a[1],:,:]
    B = np.swapaxes(B,0,1)
    S = S[:a[1]]
    # Left
    Ul,Sl,Vl = np.linalg.svd(lpsi)
    Al = np.reshape(Ul,(a[0],d,-1))
    Al = Al[:,:,:a[1]]
    Al = np.swapaxes(Al,0,1)
    Bl = np.reshape(Vl,(-1,d,a[0]))
    Bl = Bl[:a[1],:,:]
    Bl = np.swapaxes(Bl,0,1)
    Sl = Sl[:a[1]]
    # -----------------------------------------------------------------------------
    # Calculate Current & Density
    print(LBlocklr.shape,'ik')
    print(Al.conj().shape,'lim')
    print('ik,lim->klm')
    # PH - Include Singular Values
    tmp1 = einsum('ik,lim,m->klm',LBlocklr,Al.conj(),Sl)
    currTmp2 = einsum('klm,jnlo->kmno',tmp1,currentOp[0])
    lDensTmp2= einsum('klm,jnlo->kmno',tmp1,densityLOp[0])
    rDensTmp2= einsum('klm,jnlo->kmno',tmp1,densityROp[0])
    currTmp3 = einsum('kmno,okp,p->mnp',currTmp2,A,S)
    lDensTmp3= einsum('kmno,okp,p->mnp',lDensTmp2,A,S)
    rDensTmp3= einsum('kmno,okp,p->mnp',rDensTmp2,A,S)
    currTmp4 = einsum('mnp,qmr->npqr',currTmp3,Bl.conj())
    lDensTmp4= einsum('mnp,qmr->npqr',lDensTmp3,Bl.conj())
    rDensTmp4= einsum('mnp,qmr->npqr',rDensTmp3,Bl.conj())
    currTmp5 = einsum('npqr,nsqt->prst',currTmp4,currentOp[1])
    lDensTmp5= einsum('npqr,nsqt->prst',lDensTmp4,densityLOp[1])
    rDensTmp5= einsum('npqr,nsqt->prst',rDensTmp4,densityROp[1])
    currTmp6 = einsum('prst,tpu->ru',currTmp5,B)
    lDensTmp6= einsum('prst,tpu->ru',lDensTmp5,B)
    rDensTmp6= einsum('prst,tpu->ru',rDensTmp5,B)
    curr = einsum('ru,ru->',currTmp6,RBlocklr)
    lDens= einsum('ru,ru->',lDensTmp6,RBlocklr)
    rDens= einsum('ru,ru->',rDensTmp6,RBlocklr)
    #current = np.einsum('ik,lim,jnlo,okp,qmr,nsqt,tpu,ru->',LBlocklr,Al.conj(),currentOp[0],A,Bl.conj(),currentOp[1],B,RBlocklr)
    print(curr,lDens,rDens)
    # -----------------------------------------------------------------------------
    # Store left and right environments
    LBlock = einsum('ij,kil,kim->lm',LBlock,A.conj(),A)
    RBlock = einsum('ijk,ilm,km->jl',B.conj(),B,RBlock)
    LHBlock= einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[2],A)
    RHBlock= einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[2],B,RHBlock)
    # Left
    LBlockl = einsum('ij,kil,kim->lm',LBlockl,Al.conj(),Al)
    RBlockl = einsum('ijk,ilm,km->jl',Bl.conj(),Bl,RBlockl)
    LHBlockl= einsum('ijk,lim,jnlo,okp->mnp',LHBlockl,Al.conj(),Wl[2],Al)
    RHBlockl= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),Wl[2],Bl,RHBlockl)
    # Left Right
    LBlocklr = einsum('ij,kil,kim->lm',LBlocklr,Al.conj(),A)
    RBlocklr = einsum('ijk,ilm,km->jl',Bl.conj(),B,RBlocklr)
    LHBlocklr= einsum('ijk,lim,jnlo,okp->mnp',LHBlocklr,Al.conj(),Wl[2],A)
    RHBlocklr= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),Wl[2],B,RHBlocklr)
    # ------------------------------------------------------------------------------
    # Check for convergence
    if (np.abs(E - E_prev) < tol) and (np.abs(El - Elprev) < tol):
        converged = True
    else:
        E_prev = E
        El_prev = El
        if plotConv:
            Evec.append(E)
            Elvec.append(El)
            nBondVec.append(nBond)
            ax1.cla()
            ax1.plot(nBondVec,Evec,'r.')
            ax1.plot(nBondVec,Elvec,'b.')
            ax2.cla()
            ax2.semilogy(nBondVec[:-1],np.abs(Evec[:-1]-Evec[-1]),'r.')
            ax2.semilogy(nBondVec[:-1],np.abs(Elvec[:-1]-Elvec[-1]),'r.')
            plt.pause(0.01)
