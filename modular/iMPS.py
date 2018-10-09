import numpy as np
from pyscf.lib.linalg_helper import eig
from numpy import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt
import iMPO


def mpo2op():
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
                rDenMat0 = einsum('ij,jk->ik',rDenMat0,densityOp[k][:,:,i_occ[k],j_occ[k]])
                lDenMat0 = einsum('ij,jk->ik',lDenMat0,densityOp[k][:,:,i_occ[k],j_occ[k]])
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

def decomposeState():
    # Reshape wavefunction for SVD
    psi = np.reshape(psi,(2,2))
    lpsi = np.reshape(lpsi,(2,2))
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

def initializeContainers():
    # Set initial left and right containers
    LHBlock = np.array([[[1.]]])
    RHBlock = np.array([[[1.]]])
    LHBlockl= np.array([[[1.]]])
    RHBlockl= np.array([[[1.]]])
    LBlocklr = np.array([[1.]])
    RBlocklr = np.array([[1.]])
    LCurrBlock = np.array([[[1.]]])
    RCurrBlock = np.array([[[1.]]])

def evaluateOperator()
    tmp1 = einsum('ijk , lim, m->jklm',LCurrBlock, Al.conj(),   Sl.conj())
    tmp2 = einsum('jklm, jnlo  ->kmno',tmp1      , currentOp[0]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2      , A,           S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3      , Bl.conj()             )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4      , currentOp[1]          )
    tmp6 = einsum('prst, tpu   ->rsu ',tmp5      , B                     )
    curr = einsum('rsu , rsu   ->    ',tmp6      , RCurrBlock            )

def evaluateOperators():
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
    tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityOp[0]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityOp[1]          )
    tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
    denl = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )
    # Right Density ---------
    tmp1 = einsum('ik  , lim, m->klm ',LBlocklr, Al.conj(),    Sl.conj())
    tmp2 = einsum('klm , jnlo  ->kmno',tmp1,     densityOp[1]          )
    tmp3 = einsum('kmno, okp, p->mnp ',tmp2,     A,            S        )
    tmp4 = einsum('mnp , qmr   ->npqr',tmp3,     Bl.conj()              )
    tmp5 = einsum('npqr, nsqt  ->prst',tmp4,     densityOp[0]          )
    tmp6 = einsum('prst, tpu   ->su  ',tmp5,     B                      )
    denr = einsum('ru  , ru    ->    ',tmp6,     RBlocklr               )

def storeEnv():
    LHBlock= einsum('ijk,lim,jnlo,okp->mnp',LHBlock,A.conj(),W[0],A)
    RHBlock= einsum('ijk,lmin,nop,kmp->jlo',B.conj(),W[1],B,RHBlock)
    # Left
    LHBlockl= einsum('ijk,lim,jnlo,okp->mnp',LHBlockl,Al.conj(),Wl[0],Al)
    RHBlockl= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),Wl[1],Bl,RHBlockl)
    # Left Right
    LBlocklr  = einsum('jl,ijk,ilm->km',LBlocklr,Al.conj(),A)
    RBlocklr  = einsum('op,nko,nmp->km',RBlocklr,Bl.conj(),B)
    LCurrBlock= einsum('ijk,lim,jnlo,okp->mnp',LCurrBlock,Al.conj(),currentOp[0],A)
    RCurrBlock= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),currentOp[1],B,RCurrBlock)

def makeGuess():
    pass

def kernel(hamType,params,D=10,maxIter=10,tol=1e-5):
    # Get MPOs
    H = iMPO.getHam(hamType,(a,b,s))
    currOp = iMPO.currOp(hamType,(a,b,s))
    densOp = iMPO.densOp(hamType,(a,b,s))
    # 

if __name__ is "__main__":
    # Model Parameters:
    hamType = 'tasep'
    a = 0.8
    b = 2./3.
    s = 0.
    tol = 1e-5
    kernel(hamType,(a,b,s))













