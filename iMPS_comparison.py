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
gamma = 0.
delta = 0. 
p = 1.
q = 0.
s = -1.
ds = 0.001
maxBondDim = 10
maxInnerIter = 100
maxOuterIter = 50
d = 2
tol = 1e-10
plotConv = False
plotConvIn = False
hamType = 'tasep'
verbose = 1
############################################

############################################
# Determine analytic current
if (alpha > 0.5) and (beta > 0.5):
    # Maximal current phase
    J_TDL = 0.25
elif (alpha > beta):
    # High Density Phase
    J_TDL = beta*(1-beta)
elif (beta > alpha):
    # Low Density Phase
    J_TDL = alpha*(1-alpha)
else:
    # Shock Line
    J_TDL = alpha*beta
############################################

############################################
# Determine MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
W1 = []
W2 = []
if hamType == 'tasep':
    W1.insert(len(W1),np.array([[alpha*(np.exp(-(s-ds))*Sm-v),np.exp(-(s-ds))*Sp,-n,I]]))
    W1.insert(len(W1),np.array([[I],[Sm],[v],[beta*(np.exp(-(s-ds))*Sp-n)]]))
    W1.insert(len(W1),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-(s-ds))*Sp,-n,I]]))
    W2.insert(len(W2),np.array([[alpha*(np.exp(-(s+ds))*Sm-v),np.exp(-(s+ds))*Sp,-n,I]]))
    W2.insert(len(W2),np.array([[I],[Sm],[v],[beta*(np.exp(-(s+ds))*Sp-n)]]))
    W2.insert(len(W2),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-(s+ds))*Sp,-n,I]]))
elif hamType == 'sep':
    exp1_alpha = np.exp(-(s-ds))*alpha
    exp1_beta = np.exp(-(s-ds))*beta
    exp1_p = np.exp(-(s-ds))*p
    exp1_q = np.exp(s-ds)*q
    exp1_delta = np.exp(s-ds)*delta
    exp1_gamma = np.exp(s-ds)*gamma
    exp2_alpha = np.exp(-(s+ds))*alpha
    exp2_beta = np.exp(-(s+ds))*beta
    exp2_p = np.exp(-(s+ds))*p
    exp2_q = np.exp(s+ds)*q
    exp2_delta = np.exp(s+ds)*delta
    exp2_gamma = np.exp(s+ds)*gamma
    W1.insert(len(W1),np.array([[exp1_alpha*Sm-alpha*v+exp1_gamma*Sp-gamma*n, Sp, -n, Sm,-v, I]]))
    W1.insert(len(W1),np.array([[I                                      ],
                              [exp1_p*Sm                               ],
                              [p*v                                    ],
                              [exp1_q*Sp                               ],
                              [q*n                                    ],
                              [exp1_delta*Sm-delta*v+exp1_beta*Sp-beta*n]]))
    W1.insert(len(W1),np.array([[I,        z,   z, z,  z, z],
                              [exp1_p*Sm, z,   z, z,  z, z],
                              [p*v,      z,   z, z,  z, z],
                              [exp1_q*Sp, z,   z, z,  z, z],
                              [q*n,      z,   z, z,  z, z],
                              [z,        Sp, -n, Sm,-v, I]]))
    W2.insert(len(W2),np.array([[exp2_alpha*Sm-alpha*v+exp2_gamma*Sp-gamma*n, Sp, -n, Sm,-v, I]]))
    W2.insert(len(W2),np.array([[I                                      ],
                              [exp2_p*Sm                               ],
                              [p*v                                    ],
                              [exp2_q*Sp                               ],
                              [q*n                                    ],
                              [exp2_delta*Sm-delta*v+exp2_beta*Sp-beta*n]]))
    W2.insert(len(W2),np.array([[I,        z,   z, z,  z, z],
                              [exp2_p*Sm, z,   z, z,  z, z],
                              [p*v,      z,   z, z,  z, z],
                              [exp2_q*Sp, z,   z, z,  z, z],
                              [q*n,      z,   z, z,  z, z],
                              [z,        Sp, -n, Sm,-v, I]]))
############################################

############################################
# Make Initial Unit Cell
H1 = np.zeros((2**N,2**N))
H2 = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N,dtype=int)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat1 = np.array([[1]])
        tmp_mat2 = np.array([[1]])
        for k in range(N):
            tmp_mat1 = einsum('ij,jk->ik',tmp_mat1,W1[k][:,:,i_occ[k],j_occ[k]])
            tmp_mat2 = einsum('ij,jk->ik',tmp_mat2,W2[k][:,:,i_occ[k],j_occ[k]])
        H1[i,j] += tmp_mat1[[0]]
        H2[i,j] += tmp_mat2[[0]]
# Diagonalize Hamiltonian
e1,lwf1,rwf1 = la.eig(H1,left=True)
inds = np.argsort(e1)
e1 = e1[inds[-1]]
rwf1 = rwf1[:,inds[-1]]
lwf1 = lwf1[:,inds[-1]]
e2,lwf2,rwf2 = la.eig(H2,left=True)
inds = np.argsort(e2)
e2 = e2[inds[-1]]
rwf2 = rwf2[:,inds[-1]]
lwf2 = lwf2[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf1 = rwf1/np.sum(rwf1)
lwf1 = lwf1/np.sum(lwf1*rwf1)
rwf2 = rwf2/np.sum(rwf2)
lwf2 = lwf2/np.sum(lwf2*rwf2)
if verbose > 1:
    print('\nExact Diagonalization Energy: {},{},{}'.format(e1,e2,(e1-e2)/(2*ds)))
    print('Energy Check {},{}'.format(einsum('i,ij,j->',lwf1.conj(),H1,rwf1)/einsum('i,i->',lwf1.conj(),rwf1),
                                      einsum('i,ij,j->',lwf2.conj(),H2,rwf2)/einsum('i,i->',lwf2.conj(),rwf2)))
############################################

############################################
# Reshape wavefunction for SVD
rpsi1 = np.reshape(rwf1,(2,2))
lpsi1 = np.reshape(lwf1,(2,2))
rpsi2 = np.reshape(rwf2,(2,2))
lpsi2 = np.reshape(lwf2,(2,2))
if verbose > 1:
    print('After Reshaping, Energy = {},{}'.format(einsum('ij,klim,lnjo,mo->',rpsi1.conj(),W1[0],W1[1],rpsi1)/
                                                   einsum('ij,ij->',rpsi1.conj(),rpsi1),
                                                   einsum('ij,klim,lnjo,mo->',rpsi2.conj(),W2[0],W2[1],rpsi2)/
                                                   einsum('ij,ij->',rpsi2.conj(),rpsi2)))
############################################

############################################
# Do SVD of initial unit cell
Ur1,Sr1,Vr1 = np.linalg.svd(rpsi1)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A1 = np.reshape(Ur1,(a[0],d,a[1]))
A1 = np.swapaxes(A1,0,1)
B1 = np.reshape(Vr1,(a[1],d,a[0]))
B1 = np.swapaxes(B1,0,1)
Ur2,Sr2,Vr2 = np.linalg.svd(rpsi2)
a = [1,min(maxBondDim,d)] # Keep Track of bond dimensions
A2 = np.reshape(Ur2,(a[0],d,a[1]))
A2 = np.swapaxes(A2,0,1)
B2 = np.reshape(Vr2,(a[1],d,a[0]))
B2 = np.swapaxes(B2,0,1)
if verbose > 1:
    print('After SVD, Energy = {},{}'.format(einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A1.conj(),Sr1,B1.conj(),W1[0],W1[1],A1,Sr1,B1)/
                                          einsum('jik,k,lkm,jno,o,lop->',A1.conj(),Sr1,B1.conj(),A1,Sr1,B1),
                                          einsum('jik,k,lkm,nojr,oplt,rqs,s,tsu->',A2.conj(),Sr2,B2.conj(),W2[0],W2[1],A2,Sr2,B2)/
                                          einsum('jik,k,lkm,jno,o,lop->',A2.conj(),Sr2,B2.conj(),A2,Sr2,B2)))
# Store left and right environments
LBlock1 = einsum('jik,jno->ko',A1.conj(),A1)
RBlock1 = einsum('lkm,lop->ko',B1.conj(),B1)
LHBlock1= einsum('jik,nojr,rqs->kos',A1.conj(),W1[0],A1)
RHBlock1= einsum('lkm,oplt,tsu->kos',B1.conj(),W1[1],B1)
E1 = einsum('ijk,i,k,ijk->',LHBlock1,Sr1,Sr1,RHBlock1) / einsum('ko,k,o,ko->',LBlock1,Sr1,Sr1,RBlock1)
LBlock2 = einsum('jik,jno->ko',A2.conj(),A2)
RBlock2 = einsum('lkm,lop->ko',B2.conj(),B2)
LHBlock2= einsum('jik,nojr,rqs->kos',A2.conj(),W2[0],A2)
RHBlock2= einsum('lkm,oplt,tsu->kos',B2.conj(),W2[1],B2)
E2 = einsum('ijk,i,k,ijk->',LHBlock2,Sr2,Sr2,RHBlock2) / einsum('ko,k,o,ko->',LBlock2,Sr2,Sr2,RBlock2)
if verbose > 0:
    print('Energy = {},{}'.format(E1,E2))
############################################

############################################
converged = False
iterCnt = 0
nBond = 1
E1_prev = 0
E2_prev = 0
J_prev = 0
if plotConv:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
E1vec = []
E2vec = []
Jvec = []
nBondVec = []
while not converged:
    iterCnt += 1
    nBond += 2
    # Minimize energy wrt Hamiltonian
    # Increase Bond Dimension
    a[0] = a[1]
    a[1] = min(maxBondDim,a[0]*2)
    # Put previous result into initial guess
    n1,n2,n3 = A1.shape
    wfn1_ls = np.pad(A1,((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
    wfn1_rs = np.pad(B1,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
    wfn2_ls = np.pad(A2,((0,0),(0,a[0]-n2),(0,a[1]-n3)),'constant')
    wfn2_rs = np.pad(B2,((0,0),(0,a[1]-n3),(0,a[0]-n2)),'constant')
    # Jump between left and right site optimizations
    inner_converged = False
    innerIterCnt = 0
    E1prev_in = E1
    E2prev_in = E2
    while not inner_converged:
        innerIterCnt += 1
        # Push Gauge to left site --------------------------------------------
        M1_reshape = np.swapaxes(wfn1_rs,0,1)
        (n1,n2,n3) = M1_reshape.shape
        M1_reshape = np.reshape(M1_reshape,(n1,n2*n3))
        (U1,s1,V1) = np.linalg.svd(M1_reshape,full_matrices=False)
        M1_reshape = np.reshape(V1,(n1,n2,n3))
        wfn1_rs = np.swapaxes(M1_reshape,0,1)
        wfn1_ls = einsum('klj,ji,i->kli',wfn1_ls,U1,s1)
        M2_reshape = np.swapaxes(wfn2_rs,0,1)
        (n1,n2,n3) = M2_reshape.shape
        M2_reshape = np.reshape(M2_reshape,(n1,n2*n3))
        (U2,s2,V2) = np.linalg.svd(M2_reshape,full_matrices=False)
        M2_reshape = np.reshape(V2,(n1,n2,n3))
        wfn2_rs = np.swapaxes(M2_reshape,0,1)
        wfn2_ls = einsum('klj,ji,i->kli',wfn2_ls,U2,s2)
        # Calculate Inner F Block
        tmp_sum1 = einsum('cdf,eaf->acde',RHBlock1,wfn1_rs)
        tmp_sum2 = einsum('ydbe,acde->abcy',W1[2],tmp_sum1)
        F1 = einsum('bxc,abcy->xya',np.conj(wfn1_rs),tmp_sum2)
        tmp_sum1 = einsum('cdf,eaf->acde',RHBlock2,wfn2_rs)
        tmp_sum2 = einsum('ydbe,acde->abcy',W2[2],tmp_sum1)
        F2 = einsum('bxc,abcy->xya',np.conj(wfn2_rs),tmp_sum2)
        # Create Function to give Hx
        def opt_fun1(x):
            x_reshape = np.reshape(x,wfn1_ls.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',F1,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W1[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',LHBlock1,in_sum2)
            return -np.reshape(fin_sum,-1)
        def opt_fun2(x):
            x_reshape = np.reshape(x,wfn2_ls.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',F2,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W2[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',LHBlock2,in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0):
            return dx
        # Solve Eigenvalue Problem w/ Davidson Algorithm
        init_guess1 = np.reshape(wfn1_ls,-1)
        u1,v1 = eig(opt_fun1,init_guess1,precond,tol=tol)
        E1 = u1/nBond
        init_guess2 = np.reshape(wfn2_ls,-1)
        u2,v2 = eig(opt_fun2,init_guess2,precond,tol=tol)
        E2 = u2/nBond
        if verbose > 1:
            print('\tEnergy at Left Site = {},{},{},{}'.format(E1,E2,(E2-E1)/(2.*ds),J_TDL))
        wfn1_ls = np.reshape(v1,wfn1_ls.shape)
        wfn2_ls = np.reshape(v2,wfn2_ls.shape)
        # Push Gauge to right site------------------------------------------------
        (n1,n2,n3) = wfn1_ls.shape
        M1_reshape = np.reshape(wfn1_ls,(n1*n2,n3))
        (U1,s1,V1) = np.linalg.svd(M1_reshape,full_matrices=False)
        wfn1_ls = np.reshape(U1,(n1,n2,n3))
        wfn1_rs = einsum('i,ij,kjl->kil',s1,V1,wfn1_rs)
        (n1,n2,n3) = wfn2_ls.shape
        M2_reshape = np.reshape(wfn2_ls,(n1*n2,n3))
        (U2,s2,V2) = np.linalg.svd(M2_reshape,full_matrices=False)
        wfn2_ls = np.reshape(U2,(n1,n2,n3))
        wfn2_rs = einsum('i,ij,kjl->kil',s2,V2,wfn2_rs)
        # Calculate Inner f Block
        tmp_sum1 = einsum('jlp,ijk->iklp',LHBlock1,np.conj(wfn1_ls))
        tmp_sum2 = einsum('lmin,iklp->kmnp',W1[2],tmp_sum1)
        F1 = einsum('npq,kmnp->kmq',wfn1_ls,tmp_sum2)
        tmp_sum1 = einsum('jlp,ijk->iklp',LHBlock2,np.conj(wfn2_ls))
        tmp_sum2 = einsum('lmin,iklp->kmnp',W2[2],tmp_sum1)
        F2 = einsum('npq,kmnp->kmq',wfn2_ls,tmp_sum2)
        # Create Function to give Hx
        def opt_fun1(x):
            x_reshape = np.reshape(x,wfn1_rs.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',RHBlock1,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W1[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',F1,in_sum2)
            return -np.reshape(fin_sum,-1)
        def opt_fun2(x):
            x_reshape = np.reshape(x,wfn2_rs.shape)
            in_sum1 = einsum('ijk,lmk->ijlm',RHBlock2,x_reshape)
            in_sum2 = einsum('njol,ijlm->noim',W2[2],in_sum1)
            fin_sum = einsum('pnm,noim->opi',F2,in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0):
            return dx
        # Solve Eigenvalue Problem w/ Davidson Algorithm
        init_guess1 = np.reshape(wfn1_rs,-1)
        init_guess2 = np.reshape(wfn2_rs,-1)
        u1,v1 = eig(opt_fun1,init_guess1,precond,tol=tol)
        u2,v2 = eig(opt_fun2,init_guess2,precond,tol=tol)
        E1 = u1/nBond
        E2 = u2/nBond
        if verbose > 1:
            print('\tEnergy at Right Site = {},{},{},{}'.format(E1,E2,(E2-E1)/(2.*ds),J_TDL))
        wfn1_rs = np.reshape(v1,wfn1_rs.shape)
        wfn2_rs = np.reshape(v2,wfn2_rs.shape)

        if (np.abs(E1 - E1prev_in) < tol) and (np.abs(E2 - E2prev_in) < tol):
            inner_converged = True
        elif innerIterCnt > maxInnerIter:
            inner_converged = True
        else:
            if plotConvIn:
                E1prev_in = E1
                E2prev_in = E2
                E1vec.append(E1)
                E2vec.append(E2)
                Jvec.append((E2-E1)/(2.*ds))
                nBondVec.append(nBond)

                ax1.cla()
                ax1.plot(nBondVec,E1vec,'r.')
                ax1.plot(nBondVec,E2vec,'b.')
                ax1.plot(nBondVec,Jvec,'k.')

                ax2.cla()
                ax2.semilogy(nBondVec[:-1],np.abs(E1vec[:-1]-E1vec[-1]),'r.')
                ax2.semilogy(nBondVec[:-1],np.abs(E2vec[:-1]-E2vec[-1]),'b.')
                ax2.semilogy(nBondVec[:-1],np.abs(np.array(Jvec[:-1])-J_TDL),'k.')
                plt.pause(0.01)
    # -----------------------------------------------------------------------------
    # Push Gauge to left site
    M1_reshape = np.swapaxes(wfn1_rs,0,1)
    (n1,n2,n3) = M1_reshape.shape
    M1_reshape = np.reshape(M1_reshape,(n1,n2*n3))
    (U1,S1,V1) = np.linalg.svd(M1_reshape,full_matrices=False)
    M1_reshape = np.reshape(V1,(n1,n2,n3))
    B1 = np.swapaxes(M1_reshape,0,1)
    A1 = einsum('klj,ji->kli',wfn1_ls,U1)

    M2_reshape = np.swapaxes(wfn2_rs,0,1)
    (n1,n2,n3) = M2_reshape.shape
    M2_reshape = np.reshape(M2_reshape,(n1,n2*n3))
    (U2,S2,V2) = np.linalg.svd(M2_reshape,full_matrices=False)
    M2_reshape = np.reshape(V2,(n1,n2,n3))
    B2 = np.swapaxes(M2_reshape,0,1)
    A2 = einsum('klj,ji->kli',wfn2_ls,U2)

    # -----------------------------------------------------------------------------
    # Store left and right environments
    LBlock1 = einsum('ij,kil,kim->lm',LBlock1,A1.conj(),A1)
    RBlock1 = einsum('ijk,ilm,km->jl',B1.conj(),B1,RBlock1)
    LHBlock1= einsum('ijk,lim,jnlo,okp->mnp',LHBlock1,A1.conj(),W1[2],A1)
    RHBlock1= einsum('ijk,lmin,nop,kmp->jlo',B1.conj(),W1[2],B1,RHBlock1)

    LBlock2 = einsum('ij,kil,kim->lm',LBlock2,A2.conj(),A2)
    RBlock2 = einsum('ijk,ilm,km->jl',B2.conj(),B2,RBlock2)
    LHBlock2= einsum('ijk,lim,jnlo,okp->mnp',LHBlock2,A2.conj(),W2[2],A2)
    RHBlock2= einsum('ijk,lmin,nop,kmp->jlo',B2.conj(),W2[2],B2,RHBlock2)

    # ------------------------------------------------------------------------------
    # Check for convergence
    if (np.abs(E1 - E1_prev) < tol) and (np.abs(E2 - E2_prev) < tol):
        converged = True
    elif iterCnt >= maxOuterIter:
        converged = True
    else:
        E1_prev = E1
        E2_prev = E2
        E1vec.append(E1)
        E2vec.append(E2)
        Jvec.append((E2-E1)/(2.*ds))
        if verbose > 0:
            print('Energy ({})= {},{},{},{}'.format(nBond,E1,E2,(E2-E1)/(2.*ds),J_TDL))
        if plotConv:
            nBondVec.append(nBond)
            ax1.cla()
            ax1.plot(nBondVec,E1vec,'r.')
            ax1.plot(nBondVec,E2vec,'b.')
            ax1.plot(nBondVec,Jvec,'k.')
            ax2.cla()
            ax2.semilogy(nBondVec[:-1],np.abs(E1vec[:-1]-E1vec[-1]),'r.')
            ax2.semilogy(nBondVec[:-1],np.abs(E2vec[:-1]-E2vec[-1]),'b.')
            ax2.semilogy(nBondVec[:-1],np.abs(np.array(Jvec[:-1])-J_TDL),'k.')
            plt.pause(0.01)
for cnter in range(len(Jvec)):
    print(Jvec[cnter])
