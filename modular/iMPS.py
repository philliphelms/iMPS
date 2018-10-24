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

def createInitMPS(W,Wl=None,maxBondDim=10,d=2,obsvs=None):
    left=True
    if Wl is None: left=False
    H = mpo2mat([W[0],W[1]])
    # Diagonalize Hamiltonian
    (e0,lwf,rwf) = sortEig(H)
    # Ensure Proper Normalization
    # <-|R> = 1
    # <L|R> = 1
    print(e0)
    rwf = rwf/np.sum(rwf)
    lwf = lwf/np.sum(lwf*rwf)
    ############################################
    # Reshape wavefunction for SVD
    rpsi = np.reshape(rwf,(2,2))
    lpsi = np.reshape(lwf,(2,2))
    ############################################
    # Do SVD of initial unit cell
    a = [1,min(maxBondDim,d)]
    (A,S,B) = decompose(rpsi,a)
    if left: (Al,Sl,Bl) = decompose(lpsi,a)
    ############################################
    # Evaluate Observables
    if left:
        obVals = evaluateObservables([A,S,B],[Al,Sl,Bl],obsvs,init=True)
    else:
        obVals = evaluateObservables([A,S,B],[A,S,B],obsvs,init=True)
    ############################################
    # Store left and right environments
    block = makeBlocks([A,B])
    hBlock = makeBlocks([A,B],mpo=[W[0],W[1]])
    nextGuess = makeNextGuess(A,S,B,a,maxBondDim)
    E = einsum('ijk,i,k,ijk->',hBlock[0],S,S,hBlock[1]) / einsum('ko,k,o,ko->',block[0],S,S,block[1])
    if left:
        blockL  = makeBlocks([Al,Bl])
        hBlockL = makeBlocks([Al,Bl],mpo=[Wl[0],Wl[1]])
        blockLR = makeBlocks([A,B],lmps=[Al,Bl])
        hBlockLR= makeBlocks([A,B],lmps=[Al,Bl],mpo=[W[0],W[1]])
        nextGuessL = makeNextGuess(Al,Sl,Bl,a,maxBondDim)
        block = [block, blockL, blockLR ]
        hBlock= [hBlock,hBlockL,hBlockLR]
        nextGuess = [nextGuess,nextGuessL]
        El = einsum('ijk,i,k,ijk->',hBlockL[0] ,Sl,Sl,hBlockL[1] ) / einsum('ko,k,o,ko->',blockL[0] ,Sl,Sl,blockL[1] )
        Elr= einsum('ijk,i,k,ijk->',hBlockLR[0],Sl,S ,hBlockLR[1]) / einsum('ko,k,o,ko->',blockLR[0],Sl,S ,blockLR[1])
    return (E,[A,B],block,hBlock,nextGuess)

def evaluateObservables(state,lstate,obsvs,block=[np.array([[1.]]),np.array([[1.]])],init=False,norm=1.):
    for ob in obsvs:
        if ob["useBlock"] == False:
            if len(ob["mpo"]) == 2:
                tmp1 = einsum('ik  , lim, m->klm ',block[0],  lstate[0].conj(), lstate[1].conj())
                tmp2 = einsum('klm , jnlo  ->kmno',tmp1,          ob["mpo"][0]                  )
                tmp3 = einsum('kmno, okp, p->mnp ',tmp2,              state[0],         state[1])
                tmp4 = einsum('mnp , qmr   ->npqr',tmp3,      lstate[2].conj()                  )
                tmp5 = einsum('npqr, nsqt  ->prt ',tmp4,          ob["mpo"][1]                  )
                tmp6 = einsum('prt , tpu   ->ru  ',tmp5,              state[2]                  )
                ob["val"] = einsum('ru  , ru    ->    ',tmp6,              block[1]             )/norm
            else:
                ob["val"] = [None]*2
                tmp1 = einsum('ik  , lim, m->klm ',block[0],  lstate[0].conj(), lstate[1].conj())
                tmp2 = einsum('klm , jnlo  ->kmo ',tmp1,             ob["mpo"]                  )
                tmp3 = einsum('kmo , okp, p->mp  ',tmp2,              state[0],         state[1])
                tmp4 = einsum('mp  , qmr   ->pqr ',tmp3,      lstate[2].conj()                  )
                tmp5 = einsum('pqr , qpu   ->ru  ',tmp4,              state[2]                  )
                ob["val"][0] = einsum('ru  , ru    ->    ',tmp5,              block[1]          )/norm
                tmp1 = einsum('ik  , lim, m->klm ',block[0],  lstate[0].conj(), lstate[1].conj())
                tmp2 = einsum('klm , lkn, n->mn  ',tmp1    ,          state[0],         state[1])
                tmp3 = einsum('mn  , omp   ->nop ',tmp2    ,  lstate[2].conj()                  )
                tmp4 = einsum('nop , qros  ->psn ',tmp3    ,         ob["mpo"]                  )
                tmp5 = einsum('psn , snt   ->pt  ',tmp4    ,          state[2]                  )
                ob["val"][1] = einsum('pt,pt->',tmp5,block[1])/norm
        else:
            # Select correct site operators
            if init:
                mpo = [ob["mpo"][0],ob["mpo"][1]]
                newBlock = makeBlocks([state[0],state[2]],mpo,block=None,lmps=[lstate[0],lstate[2]])
            else:
                mpo = [ob["mpo"][2],ob["mpo"][2]]
                newBlock = makeBlocks([state[0],state[2]],mpo,block=ob["block"],lmps=[lstate[0],lstate[2]])
            ob["block"][0] = newBlock[0]
            ob["block"][1] = newBlock[1]
            # Evaluate Operator
            ob["val"] = einsum('ijk,ijk,i,k',ob["block"][0],ob["block"][1],state[1],lstate[1])/norm

def normalizeOpVals(obsvs,normFactor):
    for ob in obsvs:
        if ob["useBlock"] == False:
            if len(ob["mpo"]) == 2:
                ob["val"] /= normFactor
            else:
                ob["val"][0] /= normFactor
                ob["val"][1] /= normFactor
        else:
            ob["val"] /= normFactor
        if ob["print"]:
                print('\t\t'+ob["name"]+' = '+'{}'.format(ob["val"]))
    return obsvs

def makeBlocks(mps,mpo=None,block=None,lmps=None):
    if block is None:
        if mpo is None:
            block = [np.array([[1.]]),np.array([[1.]])]
        else:
            block = [np.array([[[1.]]]),np.array([[[1.]]])]
    if lmps is None:
        lmps = mps
    if mpo is None:
        tmp1 = einsum('jl,ijk->ilk',block[0],lmps[0].conj())
        block[0] = einsum('ilk,ilm->km',tmp1,mps[0])
        tmp1 = einsum('op,nko->nkp',block[1],lmps[1].conj())
        block[1] = einsum('nkp,nmp->km',tmp1,mps[1])
    else:
        tmp1 = einsum('ijk,lim->jklm',block[0],lmps[0].conj())
        tmp2 = einsum('jklm,jnlo->kmno',tmp1,mpo[0])
        block[0] = einsum('kmno,okp->mnp',tmp2,mps[0])
        tmp1 = einsum('nop,kmp->kmno',mps[1],block[1])
        tmp2 = einsum('kmno,lmin->iklo',tmp1,mpo[1])
        block[1] = einsum('iklo,ijk->jlo',tmp2,lmps[1].conj())
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
    # PH - Do for left Eigenvec
    H = einsum('ijk,jlmn,lopq,ros->mpirnqks',HBlock[0],mpo,mpo,HBlock[1])
    (n1,n2,n3,n4,n5,n6,n7,n8) = H.shape
    H = np.reshape(H,(n1*n2*n3*n4,n5*n6*n7*n8))
    return H

def setupEigenProb(mpo,HBlock,nextGuess):
    # PH - Do for left Eigenvec
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
    # PH - Do for left eigenvec
    u,v = sortEig(H,left=False)
    return u[0],v

def runEigenSolver(H):
    # PH - Do for left eigenvec
    u,v = eig(H[0],H[1],H[2])
    return -u,v

def calcEntanglement(S):
    # PH - Calc left entanglement
    entSpect = -S**2*np.log2(S**2)
    #print('\t\tEntanglement Spec: {}'.format(entSpect))
    for i in range(len(entSpect)):
        if np.isnan(entSpect[i]): entSpect[i] = 0
    entEntr = np.sum(entSpect)
    return entEntr,entSpect

def runOptR(mps,mpo,block,hBlock,nextGuess,E_init=0,maxBondDim=10,minIter=10,maxIter=10000,tol=1e-10,plotConv=True,d=2,obsvs=None):
    print('Running R Optimization Scheme')
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
        print(np.abs(E-E_prev),tol)
        if (np.abs(E - E_prev) < tol) and (iterCnt > minIter):
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

def normEigVecs(v,vl):
    v /= np.sum(v)
    vl /= np.dot(v,vl)
    return v,vl

def runOptLR(mps,mpo,block,hBlock,nextGuess,E_init=0,maxBondDim=10,minIter=10,maxIter=10000,tol=1e-10,plotConv=True,d=2,obsvs=None):
    print('Running LR Optimization Scheme')
    # Extract Inputs
    mps,mpsl = mps[0],mps[1]
    mpo,mpol = mpo[0],mpo[1]
    blockL   = block[1]
    blockLR  = block[2]
    block    = block[0]
    hBlockL  = hBlock[1]
    hBlockLR =hBlock[2]
    hBlock   = hBlock[0]
    nextGuessL = nextGuess[1]
    nextGuess = nextGuess[0]
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
        Hl = setupEigenProb(mpol,hBlockL,nextGuessL)
        El,vl = runEigenSolver(Hl)
        El /= nBond
        v,vl = normEigVecs(v,vl)
        # ------------------------------------------------------------------------------
        # Reshape result into state
        (_,_,n1,_) = mpo.shape
        (_,_,n2,_) = mpo.shape
        (n3,_,_) = hBlock[0].shape
        (n4,_,_) = hBlock[1].shape
        psi = np.reshape(v,(n1,n2,n3,n4)) # s_l s_(l+1) a_(l-1) a_(l+1)
        psi = np.transpose(psi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
        psi = np.reshape(psi,(n3*n1,n4*n2))
        lpsi= np.reshape(vl,(n1,n2,n3,n4))
        lpsi= np.transpose(lpsi,(2,0,1,3))
        lpsi= np.reshape(lpsi,(n3*n1,n4*n2))
        # ------------------------------------------------------------------------------
        # Perform USV Decomposition
        (A,S,B) = decompose(psi,a)
        mps = [A,B]
        EE,_ = calcEntanglement(S)
        (Al,Sl,Bl) = decompose(lpsi,a)
        mpsl= [Al,Bl]
        ############################################
        # Evaluate Observables
        obVals = evaluateObservables([A,S,B],[Al,Sl,Bl],block=blockLR,obsvs=obsvs)
        # -----------------------------------------------------------------------------
        # Store left and right environments
        block = makeBlocks(mps,block=block)
        hBlock = makeBlocks(mps,mpo=[mpo,mpo],block=hBlock)
        blockL = makeBlocks(mpsl,block=blockL)
        hBlockL= makeBlocks(mpsl,mpo=[mpol,mpol],block=hBlockL)
        blockLR= makeBlocks(mps,lmps=mpsl,block=blockLR)
        hBlockLR= makeBlocks(mps,lmps=mpsl,mpo=[mpo,mpo],block=hBlockLR)
        E = einsum('ijk,i,k,ijk->',hBlock[0],S,S,hBlock[1]) / einsum('ko,k,o,ko->',block[0],S,S,block[1]) / nBond
        El = einsum('ijk,i,k,ijk->',hBlockL[0] ,Sl,Sl,hBlockL[1] ) / einsum('ko,k,o,ko->',blockL[0] ,Sl,Sl,blockL[1] ) / nBond
        Elr= einsum('ijk,i,k,ijk->',hBlockLR[0],S ,Sl,hBlockLR[1]) / einsum('ko,k,o,ko->',blockLR[0],S ,Sl,blockLR[1]) / nBond
        obsvs = normalizeOpVals(obsvs,einsum('ko,k,o,ko->',blockLR[0],S ,Sl,blockLR[1]))
        # -----------------------------------------------------------------------------
        # Make next Initial Guess
        nextGuess = makeNextGuess(A,S,B,a,maxBondDim)
        nextGuessL = makeNextGuess(Al,Sl,Bl,a,maxBondDim)
        # ------------------------------------------------------------------------------
        # Check for convergence
        print('\tEnergy from Optimization = {},{},{}\tvonNeumann Entropy = {}'.format(E,El,Elr,EE))
        if (np.abs(E - E_prev) < tol) and (iterCnt > minIter):
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

def runOpt(mps,mpo,block,hBlock,nextGuess,E_init=0,maxBondDim=10,minIter=10,maxIter=10000,tol=1e-10,plotConv=True,d=2,obsvs=None):
    if len(mpo) == 2:
        E = runOptLR(mps,mpo,block,hBlock,nextGuess,E_init,maxBondDim,minIter,maxIter,tol,plotConv,d,obsvs)
    else:
        E = runOptR(mps,mpo,block,hBlock,nextGuess,E_init,maxBondDim,minIter,maxIter,tol,plotConv,d,obsvs)
    return E

if __name__ == "__main__":
    ############################################
    # Inputs
    alpha = 0.5
    beta = 0.5
    p = 1.
    s = -1.
    ds = 0.01
    hamType = 'tasep'
    mbd = 10
    ############################################
    # Run Current Calculation 1
    hmpo = createHamMPO(hamType,(alpha,beta,s))
    hmpol= createHamMPO(hamType,(alpha,beta,s),conjTrans=True)
    currGlOp = {"mpo": createGlobalCurrMPO(hamType,(alpha,beta,s)),"useBlock":True, "block":[None]*2,"print":True,"name":"Global Current","val":None}
    currLoOp = {"mpo": createLocalCurrMPO( hamType,(alpha,beta,s)),"useBlock":False,"block":[None]*2,"print":True,"name":"Local Current","val":None}
    densOp   = {"mpo": createLocalDensMPO(),                       "useBlock":False,"block":[None]*2,"print":True,"name":"Local Density","val":None}
    obsvs = [currGlOp,currLoOp,densOp]
    (E,mps,block,hBlock,nextGuess) = createInitMPS(hmpo,Wl=hmpol,obsvs=obsvs)
    E0 = runOpt(mps,[hmpo[2],hmpol[2]],block,hBlock,nextGuess,E_init=E,maxBondDim=mbd,obsvs=obsvs)
