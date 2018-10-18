import numpy as np
from pyscf.lib.linalg_helper import eig
from pyscf.lib.numpy_helper import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt

########################################################################################
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
########################################################################################

########################################################################################
# Determine MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
Wr = []
Wl = []
if hamType == 'tasep':
    Wr.insert(len(Wr),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
    Wr.insert(len(Wr),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
    Wr.insert(len(Wr),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
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
Wl.insert(len(Wl),np.transpose(Wr[0],(0,1,3,2)).conj())
Wl.insert(len(Wl),np.transpose(Wr[1],(0,1,3,2)).conj())
Wl.insert(len(Wl),np.transpose(Wr[2],(0,1,3,2)).conj())
########################################################################################

########################################################################################
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
            tmp_mat0 = einsum('ij,jk->ik',tmp_mat0,Wr[k][:,:,i_occ[k],j_occ[k]])
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
########################################################################################

########################################################################################
# Reshape wavefunction for SVD
rpsi = np.reshape(rwf,(2,2))
lpsi = np.reshape(lwf,(2,2))
print('After Reshaping, Energy (right) = {}'.format(einsum('ij,klim,lnjo,mo->',rpsi.conj(),Wr[0],Wr[1],rpsi)/
                                            einsum('ij,ij->',rpsi.conj(),rpsi)))
print('After Reshaping, Energy (left)  = {}'.format(einsum('ij,klim,lnjo,mo->',lpsi.conj(),Wl[0],Wl[1],lpsi)/
                                            einsum('ij,ij->',lpsi.conj(),lpsi)))
########################################################################################

########################################################################################
# Do SVD of initial unit cell
Ur,Sr,Vr = np.linalg.svd(rpsi)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
Ar = np.reshape(Ur,(a[0],d,a[1]))
Ar = np.swapaxes(Ar,0,1)
Br = np.reshape(Vr,(a[1],d,a[0]))
Br = np.swapaxes(Br,0,1)
print('After SVD, Energy (right) = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',Ar.conj(),Sr,Br.conj(),Wr[0],Wr[1],Ar,Sr,Br)/
                                              einsum('jik,k,lkm,jno,o,lop->',Ar.conj(),Sr,Br.conj(),Ar,Sr,Br)))
Ul,Sl,Vl = np.linalg.svd(lpsi)
Al = np.reshape(Ul,(a[0],d,a[1]))
Al = np.swapaxes(Al,0,1)
Bl = np.reshape(Vl,(a[1],d,a[0]))
Bl = np.swapaxes(Bl,0,1)
print('After SVD, Energy (left)  = {}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',Al.conj(),Sl,Bl.conj(),Wl[0],Wl[1],Al,Sl,Bl)/
                                              einsum('jik,k,lkm,jno,o,lop->',Al.conj(),Sl,Bl.conj(),Al,Sl,Bl)))
#########################################################################################

#########################################################################################
# Store left and right environments
LBlock_r = einsum('jik,jno->ko',Ar.conj(),Ar)
RBlock_r = einsum('lkm,lop->ko',Br.conj(),Br)
LHBlock_r= einsum('jik,nojr,rqs->kos',Ar.conj(),Wr[0],Ar)
RHBlock_r= einsum('lkm,oplt,tsu->kos',Br.conj(),Wr[1],Br)
Er = einsum('ijk,i,k,ijk->',LHBlock_r,Sr,Sr,RHBlock_r) / einsum('ko,k,o,ko->',LBlock_r,Sr,Sr,RBlock_r)
print('Energy (right) = {}'.format(Er))
LBlock_l = einsum('jik,jno->ko',Al.conj(),Al)
RBlock_l = einsum('lkm,lop->ko',Bl.conj(),Bl)
LHBlock_l= einsum('jik,nojr,rqs->kos',Al.conj(),Wl[0],Al)
RHBlock_l= einsum('lkm,oplt,tsu->kos',Bl.conj(),Wl[1],Bl)
El = einsum('ijk,i,k,ijk->',LHBlock_l,Sl,Sl,RHBlock_l) / einsum('ko,k,o,ko->',LBlock_l,Sl,Sl,RBlock_l)
print('Energy (left)  = {}'.format(El))
########################################################################################

########################################################################################
converged = False
iterCnt = 0
nBond = 1
Er_prev = 0
El_prev = 0
if plotConv:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
Ervec = []
Elvec = []
nBondVec = []
while not converged:
    nBond += 2
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # -----------------------------------------------------------------------------
    # Determine Hamiltonian
    Hr = einsum('ijk,jlmn,lopq,ros->mpirnqks',LHBlock_r,Wr[2],Wr[2],RHBlock_r)
    (n1,n2,n3,n4,n5,n6,n7,n8) = Hr.shape
    Hr = np.reshape(Hr,(n1*n2*n3*n4,n5*n6*n7*n8))
    # Left
    Hl = einsum('ijk,jlmn,lopq,ros->mpirnqks',LHBlock_l,Wl[2],Wl[2],RHBlock_l)
    (n1,n2,n3,n4,n5,n6,n7,n8) = Hl.shape
    Hl = np.reshape(Hl,(n1*n2*n3*n4,n5*n6*n7*n8))
    # -----------------------------------------------------------------------------
    # Solve Eigenproblem
    ur,vr = la.eig(Hr)
    ind = np.argsort(ur)[-1]
    Er = ur[ind]/nBond
    vr = vr[:,ind]
    print('\tEnergy from Optimization = {}'.format(Er))
    # Left
    ul,vl = la.eig(Hl)
    ind = np.argsort(ul)[-1]
    El = ul[ind]/nBond
    vl = vl[:,ind]
    print('\tEnergy from Optimization = {}'.format(El))
    # ------------------------------------------------------------------------------
    # Reshape result into state
    rpsi = np.reshape(vr,(n1,n2,n3,n4)) # s_l s_(l+1) a_(l-1) a_(l+1)
    rpsi = np.transpose(rpsi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
    rpsi = np.reshape(rpsi,(n3*n1,n4*n2))
    # Left
    lpsi = np.reshape(vl,(n1,n2,n3,n4)) # s_l s_(l+1) a_(l-1) a_(l+1)
    lpsi = np.transpose(lpsi,(2,0,1,3)) # a_(l-1) s_l a_(l+1) s_(l+1)
    lpsi = np.reshape(lpsi,(n3*n1,n4*n2))
    # ------------------------------------------------------------------------------
    # Canonicalize state
    Ur,Sr,Vr = np.linalg.svd(rpsi)
    Ar = np.reshape(Ur,(a[0],d,-1))
    Ar = Ar[:,:,:a[1]]
    Ar = np.swapaxes(Ar,0,1)
    Br = np.reshape(Vr,(-1,d,a[0]))
    Br = Br[:a[1],:,:]
    Br = np.swapaxes(Br,0,1)
    Sr = Sr[:a[1]]
    # Left
    Ul,Sl,Vl = np.linalg.svd(lpsi)
    Al = np.reshape(Ul,(a[0],d,-1))
    Al = Al[:,:,:a[1]]
    Al = np.swapaxes(Al,0,1)
    Bl = np.reshape(Vl,(-1,d,a[0]))
    Bl = Bl[:a[1],:,:]
    Bl = np.swapaxes(Bl,0,1)
    Sl = Sl[:a[1]]
    #E = einsum('ijk,lim,jnlo,okp,qmr,nsqt,tpu,rsu,m,p->',LHBlock,A.conj(),W[2],A,B.conj(),W[2],B,RHBlock,S,S)/nBond
    #print('\tEnergy after SVD = {}'.format(E))
    # -----------------------------------------------------------------------------
    # Store left and right environments
    LBlock_r = einsum('ij,kil,kim->lm',LBlock_r,Ar.conj(),Ar)
    RBlock_r = einsum('ijk,ilm,km->jl',Br.conj(),Br,RBlock_r)
    LHBlock_r= einsum('ijk,lim,jnlo,okp->mnp',LHBlock_r,Ar.conj(),Wr[2],Ar)
    RHBlock_r= einsum('ijk,lmin,nop,kmp->jlo',Br.conj(),Wr[2],Br,RHBlock_r)
    num = einsum('ijk,i,k,ijk->',LHBlock_r,Sr,Sr,RHBlock_r)
    den = einsum('ko,k,o,ko->',LBlock_r,Sr,Sr,RBlock_r)
    # Left
    LBlock_l = einsum('ij,kil,kim->lm',LBlock_l,Al.conj(),Al)
    RBlock_l = einsum('ijk,ilm,km->jl',Bl.conj(),Bl,RBlock_l)
    LHBlock_l= einsum('ijk,lim,jnlo,okp->mnp',LHBlock_l,Al.conj(),Wl[2],Al)
    RHBlock_l= einsum('ijk,lmin,nop,kmp->jlo',Bl.conj(),Wl[2],Bl,RHBlock_l)
    num = einsum('ijk,i,k,ijk->',LHBlock_l,Sl,Sl,RHBlock_l)
    den = einsum('ko,k,o,ko->',LBlock_l,Sl,Sl,RBlock_l)
    #E = einsum('ijk,i,k,ijk->',LHBlock,S,S,RHBlock) / einsum('ko,k,o,ko->',LBlock,S,S,RBlock)/nBond
    #print('\tEnergy after storing Blocks = {}'.format(E))
    # ------------------------------------------------------------------------------
    # Check for convergence
    if (np.abs(Er - Er_prev) < tol) and (np.abs(El - El_prev) < tol):
        converged = True
    else:
        Er_prev = Er
        El_prev = El
        if plotConv:
            Ervec.append(Er)
            Elvec.append(El)
            nBondVec.append(nBond)
            ax1.cla()
            ax1.plot(nBondVec,Ervec,'r.')
            ax1.plot(nBondVec,Elvec,'b.')
            ax2.cla()
            ax2.semilogy(nBondVec[:-1],np.abs(Ervec[:-1]-Ervec[-1]),'r.')
            ax2.semilogy(nBondVec[:-1],np.abs(Elvec[:-1]-Elvec[-1]),'r.')
            plt.pause(0.01)
