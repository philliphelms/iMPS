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
alpha = 0.8
beta = 2./3.
gamma = 0.
delta = 0. 
p = 1.
q = 0.
s = -0.5
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
currentOp = [None]*3
currentOp[0] = np.array([[exp_alpha*Sm+exp_gamma*Sp,Sp,Sm,I]])
currentOp[1] = np.array([[I                       ],
                         [exp_p*Sm                ],
                         [exp_q*Sp                ],
                         [exp_delta*Sm+exp_beta*Sp]])
currentOp[2] = np.array([[I,        z,   z, z],
                         [exp_p*Sm, z,   z, z],
                         [exp_q*Sp, z,   z, z],
                         [z,        Sp, Sm, I]])
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
currH = np.zeros((2**2,2**2))
rDenH = np.zeros((2**2,2**2))
lDenH = np.zeros((2**2,2**2))
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
        currMat0 = np.array([[1]])
        rDenMat0 = np.array([[1]])
        lDenMat0 = np.array([[1]])
        for k in range(2):
            tmp_mat0 = einsum('ij,jk->ik',tmp_mat0,W[k][:,:,i_occ[k],j_occ[k]])
            currMat0 = einsum('ij,jk->ik',currMat0,currentOp[k][:,:,i_occ[k],j_occ[k]])
            rDenMat0 = einsum('ij,jk->ik',rDenMat0,densityROp[k][:,:,i_occ[k],j_occ[k]])
            lDenMat0 = einsum('ij,jk->ik',lDenMat0,densityLOp[k][:,:,i_occ[k],j_occ[k]])
        H[i,j]     += tmp_mat0[[0]]
        currH[i,j] += currMat0[[0]]
        rDenH[i,j] += rDenMat0[[0]]
        lDenH[i,j] += lDenMat0[[0]]
# Diagonalize Hamiltonian
u,lpsi,psi = la.eig(H,left=True)
inds = np.argsort(u)
u = u[inds[-1]]
psi = psi[:,inds[-1]]
lpsi = lpsi[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
psi = psi/np.sum(psi)
lpsi = lpsi/np.sum(lpsi*psi)
############################################

############################################
# Reshape wavefunction for SVD
psi = np.reshape(psi,(2,2))
lpsi = np.reshape(lpsi,(2,2))
############################################

############################################
# Do SVD of initial unit cell
U,S,V = np.linalg.svd(psi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A = np.reshape(U,(a[0],d,a[1]))
A = np.swapaxes(A,0,1)
B = np.reshape(V,(a[1],d,a[0]))
B = np.swapaxes(B,0,1)
# Left
Ul,Sl,Vl = np.linalg.svd(lpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
Al = np.reshape(Ul,(a[0],d,a[1]))
Al = np.swapaxes(Al,0,1)
Bl = np.reshape(Vl,(a[1],d,a[0]))
Bl = np.swapaxes(Bl,0,1)
############################################

############################################
# Set initial left and right containers
LBlock = np.array([[1.]])
RBlock = np.array([[1.]])
LHBlock = np.array([[[1.]]])
RHBlock = np.array([[[1.]]])
LBlockl = np.array([[1.]])
RBlockl = np.array([[1.]])
LHBlockl= np.array([[[1.]]])
RHBlockl= np.array([[[1.]]])
LBlocklr = np.array([[1.]])
RBlocklr = np.array([[1.]])
LHBlocklr= np.array([[[1.]]])
RHBlocklr= np.array([[[1.]]])
LCurrBlock = np.array([[[1.]]])
RCurrBlock = np.array([[[1.]]])
############################################

############################################
# Evaluate Operators
# total current ---------
tmp1 = einsum('ijk , lim, m->jklm',LCurrBlock, Al.conj(),   Sl.conj())
tmp2 = einsum('jklm, jnlo  ->kmno',tmp1      , currentOp[0]          )
tmp3 = einsum('kmno, okp, p->mnp ',tmp2      , A,           S        )
tmp4 = einsum('mnp , qmr   ->npqr',tmp3      , Bl.conj()             )
tmp5 = einsum('npqr, nsqt  ->prst',tmp4      , currentOp[1]          )
tmp6 = einsum('prst, tpu   ->rsu ',tmp5      , B                     )
curr = einsum('rsu , rsu   ->    ',tmp6      , RCurrBlock            )
# Left  Density ---------
tmp1 = einsum('ik  , lim, m->klm ',LBlocklr, Al.conj(),    Sl.conj())
tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityLOp[0]          )
tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityLOp[1]          )
tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
denl = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )
# Right Density ---------
tmp1 = einsum('ik  , lim, m->klm ',LBlocklr, Al.conj(),    Sl.conj())
tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityROp[0]          )
tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityROp[1]          )
tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
denr = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )
############################################

############################################
# Store left and right environments
LBlock = einsum('ij,kil,kim->lm',LBlock,A.conj(),A)
RBlock = einsum('ijk,ilm,km->jl',B.conj(),B,RBlock)
LHBlock= einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[0],A)
RHBlock= einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[1],B,RHBlock)
# Left
LBlockl = einsum('ij,kil,kim->lm',LBlockl,Al.conj(),Al)
RBlockl = einsum('ijk,ilm,km->jl',Bl.conj(),Bl,RBlockl)
LHBlockl= einsum('ijk,lim,jnlo,okp->mnp',LHBlockl,Al.conj(),Wl[0],Al)
RHBlockl= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),Wl[1],Bl,RHBlockl)
# Left Right
LBlocklr  = einsum('jl,ijk,ilm->km',LBlocklr,Al.conj(),A)
RBlocklr  = einsum('op,nko,nmp->km',RBlocklr,Bl.conj(),B)
LHBlocklr = einsum('ijk,lim,jnlo,okp->mnp',LHBlocklr,Al.conj(),W[0],A)
RHBlocklr = einsum('qmr,nsqt,tpu,rsu->mnp',Bl.conj(),W[1],B,RHBlocklr)
LCurrBlock= einsum('ijk,lim,jnlo,okp->mnp',LCurrBlock,Al.conj(),currentOp[0],A)
RCurrBlock= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),currentOp[1],B,RCurrBlock)
print(u,curr,denl,denr)
############################################

############################################
converged = False
iterCnt = 0
nBond = 1
E_prev = 0
El_prev = 0
curr_prev = 0
if plotConv:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
Evec = []
Elvec = []
currVec = []
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
    # Left
    ul,vl = eig(Hlx,initGuessl,precond) # PH - Add tolerance here?
    El = -u/nBond
    # PH - Figure out normalization
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
    # Total Current ---------
    tmp1 = einsum('ijk , lim, m->jklm',LCurrBlock, Al.conj(),   Sl.conj())
    tmp2 = einsum('jklm, jnlo  ->kmno',tmp1      , currentOp[2]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2      , A,           S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3      , Bl.conj()             )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4      , currentOp[2]          )
    tmp6 = einsum('prst, tpu   ->rsu ',tmp5      , B                     )
    curr = einsum('rsu , rsu   ->    ',tmp6      , RCurrBlock            )
    # Left  Density ---------
    tmp1 = einsum('ik  , lim, m->klm ',LBlocklr, Al.conj(),    Sl.conj())
    tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityLOp[0]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityLOp[1]          )
    tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
    denl = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )
    # Right Density ---------
    tmp1 = einsum('ik  , lim, m->klm ',LBlocklr, Al.conj(),    Sl.conj())
    tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityROp[0]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityROp[1]          )
    tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
    denr = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )
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
    LBlocklr  = einsum('jl,ijk,ilm->km',LBlocklr,Al.conj(),A)
    RBlocklr  = einsum('op,nko,nmp->km',RBlocklr,Bl.conj(),B)
    LHBlocklr = einsum('ijk,lim,jnlo,okp->mnp',LHBlocklr,Al.conj(),W[2],A)
    RHBlocklr = einsum('qmr,nsqt,tpu,rsu->mnp',Bl.conj(),W[2],B,RHBlocklr)
    LCurrBlock= einsum('ijk,lim,jnlo,okp->mnp',LCurrBlock,Al.conj(),currentOp[2],A)
    RCurrBlock= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),currentOp[2],B,RCurrBlock)
    # ------------------------------------------------------------------------------
    # Determine Normalization Factor
    normFact = einsum('lm,lm,l,m->',LBlocklr,RBlocklr,Sl,S)
    curr /= nBond*normFact
    denl /= normFact
    denr /= normFact
    print(normFact)
    print(E,curr,denl,denr)
    # ------------------------------------------------------------------------------
    # Check for convergence
    if (np.abs(E - E_prev) < tol) and (np.abs(El - El_prev) < tol) and (np.abs(curr-curr_prev) < tol):
        converged = True
    else:
        E_prev = E
        El_prev = El
        curr_prev = curr
        if plotConv:
            Evec.append(E)
            Elvec.append(El)
            currVec.append(curr)
            nBondVec.append(nBond)
            ax1.cla()
            ax1.plot(nBondVec,Evec,'r.')
            ax1.plot(nBondVec,Elvec,'b.')
            ax2.cla()
            ax2.semilogy(nBondVec[:-1],np.abs(Evec[:-1]-Evec[-1]),'r.')
            ax2.semilogy(nBondVec[:-1],np.abs(Elvec[:-1]-Elvec[-1]),'r.')
            plt.pause(0.01)
