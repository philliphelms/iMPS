import numpy as np
from pyscf.lib.linalg_helper import eig
from pyscf.lib.numpy_helper import einsum
from scipy import linalg as la
from iMPO import *

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


def sortEig(mat,left=True,n_eig=1):
    # Return n_eig smallest eigenvalues and associated eigenvectors
    if left:
        e0,lv,rv = la.eig(mat,left=True)
        inds = np.argsort(e0)[::-1]
        print(e0)
        e0 = e0[inds[:n_eig]]
        lv = lv[:,inds[:n_eig]]
        rv = rv[:,inds[:n_eig]]
        return (e0,lv,rv)
    else:
        e0,rv = la.eig(mat)
        inds = np.argsort(e0)[::-1]
        e0 = e0[inds[:n_eig]]
        rv = rv[:,inds[:n_eig]]
        return e0,rv

def createInitMPS(W,maxBondDim=10,d=2):
    H = mpo2mat([W[0],W[1]])
    # Diagonalize Hamiltonian
    (e0,lwf,rwf) = sortEig(H)
    # Ensure Proper Normalization
    # <-|R> = 1
    # <L|R> = 1
    rwf = rwf/np.sum(rwf)
    lwf = lwf/np.sum(lwf*rwf)
    #print('\nExact Diagonalization Energy: {}'.format(e0))
    ############################################
    # Reshape wavefunction for SVD
    rpsi = np.reshape(rwf,(2,2))
    lpsi = np.reshape(lwf,(2,2))
    ############################################
    # Do SVD of initial unit cell
    a = [1,min(maxBondDim,d)]
    (A,S,B) = decompose(rpsi,a)
    ############################################
    # Store left and right environments
    block = makeBlocks([A,B])
    hBlock = makeBlocks([A,B],mpo=[W[0],W[1]])
    E = einsum('ijk,i,k,ijk->',hBlock[0],S,S,hBlock[1]) / einsum('ko,k,o,ko->',block[0],S,S,block[1])
    ############################################
    # Create the next guess
    nextGuess = makeNextGuess(A,S,B,a,maxBondDim)
    return (E,[A,B],block,hBlock,nextGuess)

def makeBlocks(mps,mpo=None,block=None):
    if block is None:
        if mpo is None:
            block = [np.array([[1.]]),np.array([[1.]])]
        else:
            block = [np.array([[[1.]]]),np.array([[[1.]]])]
    if mpo is None:
        #print(np.sum(np.sum(np.abs(einsum('ilk,ilm->km',einsum('jl,ijk->ilk',block[0],mps[0].conj()),mps[0])-
        #                           einsum('ij,kil,kim->lm',block[0],mps[0].conj(),mps[0])))))
        #print(np.sum(np.sum(np.abs(einsum('nkp,nmp->km',einsum('op,nko->nkp',block[1],mps[1].conj()),mps[1])-
        #                           einsum('ijk,ilm,km->jl',mps[1].conj(),mps[1],block[1])))))
        tmp1 = einsum('jl,ijk->ilk',block[0],mps[0].conj())
        block[0] = einsum('ilk,ilm->km',tmp1,mps[0])
        tmp1 = einsum('op,nko->nkp',block[1],mps[1].conj())
        block[1] = einsum('nkp,nmp->km',tmp1,mps[1])
    else:
        #print(np.sum(np.sum(np.abs(einsum('kmno,okp->mnp',einsum('jklm,jnlo->kmno',einsum('ijk,lim->jklm',block[0],mps[0].conj()),mpo[0]),mps[0])-
        #                           einsum('ijk,lim,jnlo,okp->mnp',block[0],mps[0].conj(),mpo[0],mps[0])))))
        #print(np.sum(np.sum(np.abs(einsum('iklo,ijk->jlo',einsum('kmno,lmin->iklo',einsum('nop,kmp->kmno',mps[1],block[1]),mpo[1]),mps[1].conj())-
        #                           einsum('ijk,lmin,nop,kmp->jlo',mps[1].conj(),mpo[1],mps[1],block[1])))))
        tmp1 = einsum('ijk,lim->jklm',block[0],mps[0].conj())
        tmp2 = einsum('jklm,jnlo->kmno',tmp1,mpo[0])
        block[0] = einsum('kmno,okp->mnp',tmp2,mps[0])
        tmp1 = einsum('nop,kmp->kmno',mps[1],block[1])
        tmp2 = einsum('kmno,lmin->iklo',tmp1,mpo[1])
        block[1] = einsum('iklo,ijk->jlo',tmp2,mps[1].conj())
    return block

def makeNextGuess(A,S,B,a,maxBondDim=10):
    a0 = a[1]
    a1 = min(maxBondDim,a0*2)
    (n1,n2,n3) = A.shape
    Aguess = np.pad(einsum('ijk,k->ijk',A,S),((0,0),(0,a0-n2),(0,a1-n3)),'constant')
    Bguess = np.pad(B,((0,0),(0,a1-n3),(0,a0-n2)),'constant')
    initGuess = einsum('ijk,lkm->iljm',Aguess,Bguess)
    return initGuess

def decompose(psi,a,d=2):
    # Canonicalize state
    U,S,V = np.linalg.svd(psi)
    A = np.reshape(U,(a[0],d,-1))
    A = A[:,:,:a[1]]
    A = np.swapaxes(A,0,1)
    B = np.reshape(V,(-1,d,a[0]))
    B = B[:a[1],:,:]
    B = np.swapaxes(B,0,1)
    S = S[:a[1]]
    return (A,S,B)

def initializePlot(plotConv):
    if plotConv:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        return (fig,ax1,ax2)
    else: return None

def updatePlot(plotConv,f,Evec,nVec):
    if plotConv:
        f[1].cla()
        f[1].plot(nVec,Evec,'r.')
        f[2].cla()
        f[2].semilogy(nVec[:-1],np.abs(Evec[:-1]-Evec[-1]),'r.')
        plt.pause(0.01)

def setupEigenProbSlow(HBlock,mpo):
    H = einsum('ijk,jlmn,lopq,ros->mpirnqks',HBlock[0],mpo,mpo,HBlock[1])
    (n1,n2,n3,n4,n5,n6,n7,n8) = H.shape
    H = np.reshape(H,(n1*n2*n3*n4,n5*n6*n7*n8))
    return H

def setupEigenProb(mpo,HBlock,nextGuess):
    guessShape = nextGuess.shape
    def Hx(x):
        x_reshape = np.reshape(x,guessShape)
        tmp1 = einsum('ijk,nqks->ijnqs',HBlock[0],x_reshape) # Could be 'ijk,mpir->jkmpr'
        tmp2 = einsum('jlmn,ijnqs->ilmqs',mpo,tmp1)
        tmp3 = einsum('lopq,ilmqs->imops',mpo,tmp2)
        finalVec = einsum('ros,imops->mpir',HBlock[1],tmp3)
        return -finalVec.ravel()
    def precond(dx,e,x0):
        return dx
    return (Hx,nextGuess.ravel(),precond)

def runEigenSolverSlow(H):
    u,v = sortEig(H,left=False)
    return u[0],v

def runEigenSolver(H):
    u,v = eig(H[0],H[1],H[2])
    return -u,v

def calcEntanglement(S):
    entSpect = -S**2*np.log2(S**2)
    for i in range(len(entSpect)):
        if np.isnan(entSpect[i]): entSpect[i] = 0
    entEntr = np.sum(entSpect)
    return entEntr,entSpect

def runOpt(mps,mpo,block,hBlock,nextGuess,E_init=0,maxBondDim=10,maxIter=10000,tol=1e-8,plotConv=True,d=2):
    # Extract Inputs
    A,B = mps[0],mps[1]
    lBlock,rBlock = block[0],block[1]
    lHBlock,rHBlock = hBlock[0],hBlock[1]
    # Set up Iterative Loop Parameters
    fig = initializePlot(plotConv)
    converged = False
    iterCnt = 0
    nBond = 1
    E_prev = E_init
    a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
    Evec = []
    nBondVec = []
    while not converged:
        nBond += 2
        a[0] = a[1]
        a[1] = min(maxBondDim,a[0]*2)
        # ------------------------------------------------------------------------------
        # Run Eigensolver
        H = setupEigenProb(mpo,hBlock,nextGuess)
        E,v = runEigenSolver(H)
        E /= nBond
        # ------------------------------------------------------------------------------
        # Reshape result into state
        (_,_,n1,_) = mpo.shape
        (_,_,n2,_) = mpo.shape
        (n3,_,_) = hBlock[0].shape
        (n4,_,_) = hBlock[1].shape
        psi = np.reshape(v,(n1,n2,n3,n4)) # s_l s_(l+1) a_(l-1) a_(l+1)
        psi = np.transpose(psi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
        psi = np.reshape(psi,(n3*n1,n4*n2))
        # ------------------------------------------------------------------------------
        # Perform USV Decomposition
        (A,S,B) = decompose(psi,a,d=2)
        mps = [A,B]
        EE,_ = calcEntanglement(S)
        # -----------------------------------------------------------------------------
        # Store left and right environments
        block = makeBlocks(mps,block=block)
        hBlock = makeBlocks(mps,mpo=[mpo,mpo],block=hBlock)
        # -----------------------------------------------------------------------------
        # Make next Initial Guess
        nextGuess = makeNextGuess(A,S,B,a,maxBondDim)
        # ------------------------------------------------------------------------------
        # Check for convergence
        print('\tEnergy from Optimization = {}\tvonNeumann Entropy = {}'.format(E,EE))
        if np.abs(E - E_prev) < tol:
            converged = True
            print('System Converged {} {}'.format(E,E_prev))
        elif iterCnt == maxIter:
            converged = True
            print('Convergence not acheived')
        else:
            E_prev = E
            iterCnt += 1
            Evec.append(E)
            nBondVec.append(nBond)
            updatePlot(plotConv,fig,Evec,nBondVec)
    return E

if __name__ == "__main__":
    ############################################
    # Inputs
    alpha = 0.35
    beta = 2./3.
    p = 1.
    s = 10.
    ds = 0.01
    hamType = 'tasep'
    mbd = 10
    ############################################
    # Run Current Calculation 1
    hmpo = createHamMPO(hamType,(alpha,beta,s+ds))
    (E,mps,block,hBlock,nextGuess) = createInitMPS(hmpo)
    E1 = runOpt(mps,hmpo[2],block,hBlock,nextGuess,E_init=E,maxBondDim=mbd)
    # Run Current Calculation 2
    hmpo = createHamMPO(hamType,(alpha,beta,s-ds))
    (E,mps,block,hBlock,nextGuess) = createInitMPS(hmpo)
    E2 = runOpt(mps,hmpo[2],block,hBlock,nextGuess,E_init=E,maxBondDim=mbd)
    print('Current = {}'.format((E1-E2)/(2*ds)))
