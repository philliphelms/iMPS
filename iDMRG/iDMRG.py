import numpy as np
from iMPO import *
from scipy.linalg import eig

def sortEig(mat,n_eig=1):
    e0,lv,rv = eig(mat,left=True)
    inds = np.argsort(e0)[::-1]
    e0 = e0[inds[:n_eig]]
    lv = lv[:,inds[:n_eig]]
    rv = rv[:,inds[:n_eig]]
    return (e0,lv,rv)

def mpsMatDim(N,maxBondDim,d=2):
    assert(N%2==0)
    # Determine Matrix Dimensions
    fbd_site = []
    mbd_site = []
    fbd_site.insert(0,1)
    mbd_site.insert(0,1)
    for i in range(int(N/2)):
        fbd_site.insert(-1,d**i)
        mbd_site.insert(-1,min(d**i,maxBondDim))
    for i in range(int(N/2))[::-1]:
        fbd_site.insert(-1,d**(i+1))
        mbd_site.insert(-1,min(d**(i+1),maxBondDim))
    return fbd_site, mbd_site

def calcEntanglementEnt(S):
    return -np.dot(S**2.,np.log2(S**2.))

def vec2mps(N,vec,maxBondDim=100,d=2):
    fbd,mbd = mpsMatDim(N,maxBondDim,d)
    mps = []
    ee = np.zeros(N-1)
    for i in range(N,1,-1):
        vec_rshp = np.reshape(vec,(d**(i-1),-1))
        (U,S,V) = np.linalg.svd(vec_rshp,full_matrices=False)
        B = np.reshape(V,(fbd[i-1],d,mbd[i]))
        B = B[:mbd[i-1],:,:mbd[i]]
        B = np.swapaxes(B,0,1)
        mps.insert(0,B)
        vec = np.einsum('ij,j->ij',U[:,:mbd[i-1]],S[:mbd[i-1]])
        ee[i-2] = calcEntanglementEnt(S)
    mps.insert(0,np.reshape(vec,(d,1,min(d,maxBondDim))))
    return mps,ee

def moveGaugeRight(mps,site=None,sepSingVals=False):
    N = len(mps)
    if (N%2 == 1):
        mps = contractCenter(mps,'right')
        N = len(mps)
    if site is None:
        site = int(N/2)-1
    (n1,n2,n3) = mps[site].shape
    M_reshape = np.reshape(mps[site],(n1*n2,n3))
    (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
    mps[site] = np.reshape(U,(n1,n2,n3))
    if sepSingVals:
        mps.insert(site+1,S)
        mps[site+2] = np.einsum('ij,kjl->kil',V,mps[site+2])
    else:
        mps[site+1] = np.einsum('i,ij,kjl->kil',S,V,mps[site+1])
    return mps

def contractCenter(mps,direction):
    N = len(mps)
    cInd = int(N/2)
    if direction == 'left':
        mps[cInd-1] = np.einsum('ijk,k->ijk',mps[cInd-1],mps[cInd])
    else:
        mps[cInd+1] = np.einsum('ijk,j->ijk',mps[cInd+1],mps[cInd])
    mps.pop(cInd)
    return mps

def moveGaugeLeft(mps,site=None,sepSingVals=False):
    N = len(mps)
    if (N%2 == 1):
        mps = contractCenter(mps,'left')
        N = len(mps)
    if site is None:
        site = int(N/2)-1
    (n1,n2,n3) = mps[site].shape
    M_reshape = np.swapaxes(mps[site],0,1)
    M_reshape = np.reshape(M_reshape,(n2,n1*n3))
    (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(n2,n1,n3))
    mps[site] = np.swapaxes(M_reshape,0,1)
    if sepSingVals:
        mps.insert(site,S)
        mps[site-1] = np.einsum('klj,ji->kli',mps[site-1],U)
    else:
        mps[site-1] = np.einsum('klj,ji,i->kli',mps[site-1],U,S)
    return mps

def centerGauge(mps):
    N = len(mps)
    for site in range(int(N/2)-1):
        mps = moveGaugeRight(mps,site)
    mps = moveGaugeRight(mps,int(N/2)-1,sepSingVals=True)
    return mps 

def slowExactDiag(N,mpo,maxBondDim=100):
    mpoList = []
    mpoList.insert(0,mpo[1])
    for i in range(N-2):
        mpoList.insert(0,mpo[2])
    mpoList.insert(0,mpo[0])
    H = mpo2mat(mpoList)
    # Diagonalize Hamiltonian
    (e0,lwf,rwf) = sortEig(H)
    rwf = rwf/np.sum(rwf)
    lwf = lwf/np.sum(lwf*rwf)
    rmps,eer = vec2mps(N,rwf,maxBondDim)
    lmps,eel = vec2mps(N,lwf,maxBondDim)
    rmps = centerGauge(rmps)
    lmps = centerGauge(lmps)
    return rmps,lmps

def initialTwoSite(mpo):
    rmps,lmps = slowExactDiag(2,mpo)
    return rmps,lmps

def initialFourSite(mpo):
    rmps,lmps = slowExactDiag(4,mpo)
    return rmps,lmps

def calcLambdaL(mps):
    N = len(mps)
    assert(N%2 == 1)
    leftMovedMPS = moveGaugeLeft(mps,sepSingVals=True)
    lmbda = leftMovedMPS[int(N/2)-1]
    B = leftMovedMPS[int(N/2)]
    return lmbda,B

def calcLambdaR(mps):
    N = len(mps)
    assert(N%2 == 1)
    rightMovedMPS = moveGaugeRight(mps,sepSingVals=True)
    lmbda = rightMovedMPS[int(N/2)+1]
    A = rightMovedMPS[int(N/2)]
    return lmbda,A

def guessWaveFunction(A,R,inv,L,B):
    return 0

def kernel(mpo):
    # Step 1: Initialization
    rmps2,lmps2 = initialTwoSite(mpo)
    rmps4,lmps4 = initialFourSite(mpo)
    n = 1
    converged = False
    rmps,lmps = rmps4,lmps4
    while not converged:
        N = len(rmps)
        lambda_nm1 = rmps[int(N/2)].copy()
        lambdaL,B_np1 = calcLambdaL(rmps.copy())
        lambdaR,A_np1 = calcLambdaR(rmps.copy())
        initGuess = guessWavefunction(A_np1,lambdaR,np.inv(lambda_nm1),lambdaL,B_np1)
        solveEig()
        truncateMPS()
        checkConv()

if __name__ == "__main__":
    ############################################
    # Inputs
    alpha = 0.35
    beta = 2./3.
    p = 1.
    s = -1. 
    ds = 0.01
    hamType = 'tasep'
    mbd = 100 
    ############################################
    # Run Current Calculation 1
    hmpo = createHamMPO(hamType,(alpha,beta,s))
    hmpol= createHamMPO(hamType,(alpha,beta,s),conjTrans=True)
    kernel(hmpo)
    # Run Current Calculation 2
    hmpo = createHamMPO(hamType,(alpha,beta,1.))
    hmpol= createHamMPO(hamType,(alpha,beta,1.),conjTrans=True)
    kernel(hmpo)
