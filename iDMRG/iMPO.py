import numpy as np
from pyscf.lib.linalg_helper import eig
from pyscf.lib.numpy_helper import einsum
from scipy import linalg as la
import matplotlib.pyplot as plt


def createHamMPO(hamType,hamParams,conjTrans=False):
    ############################################
    # Determine MPO
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    ham = []
    if hamType == 'tasep':
        # Totally Asymmetric Simple Exclusion Process -----------------------------------------
        alpha = hamParams[0]
        beta = hamParams[1]
        s = hamParams[2]
        ham.insert(len(ham),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
        ham.insert(len(ham),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
        ham.insert(len(ham),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
    elif hamType == 'sep':
        # Generic Simple Exclusion Process -----------------------------------------------------
        alpha = float(hamParams[0])
        gamma = float(hamParams[1])
        p = float(hamParams[2])
        q = float(hamParams[3])
        beta = float(hamParams[4])
        delta = float(hamParams[5])
        s = float(hamParams[6])
        exp_alpha = np.exp(-s)*alpha
        exp_beta = np.exp(-s)*beta
        exp_p = np.exp(-s)*p
        exp_q = np.exp(s)*q
        exp_delta = np.exp(s)*delta
        exp_gamma = np.exp(s)*gamma
        ham.insert(len(ham),np.array([[exp_alpha*Sm-alpha*v+exp_gamma*Sp-gamma*n, Sp, -n, Sm,-v, I]]))
        ham.insert(len(ham),np.array([[I                                      ],
                                  [exp_p*Sm                               ],
                                  [p*v                                    ],
                                  [exp_q*Sp                               ],
                                  [q*n                                    ],
                                  [exp_delta*Sm-delta*v+exp_beta*Sp-beta*n]]))
        ham.insert(len(ham),np.array([[I,        z,   z, z,  z, z],
                                  [exp_p*Sm, z,   z, z,  z, z],
                                  [p*v,      z,   z, z,  z, z],
                                  [exp_q*Sp, z,   z, z,  z, z],
                                  [q*n,      z,   z, z,  z, z],
                                  [z,        Sp, -n, Sm,-v, I]]))
    ############################################
    # conjugate transpose operator if desired
    if conjTrans:
        for site in range(len(ham)):
            ham[site] = np.transpose(ham[site],(0,1,3,2)).conj()
    return ham

def mpo2mat(mpo):
    N = len(mpo)
    (_,_,d,_) = mpo[0].shape
    mat = np.zeros((d**N,d**N))
    occ = np.zeros((d**N,N),dtype=int)
    for i in range(d**N):
        occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[N:]))+bin(i)[N:])))
    for i in range(d**N):
        i_occ = occ[i,:]
        for j in range(d**N):
            j_occ = occ[j,:]
            tmp_mat = np.array([[1]])
            for k in range(N):
                tmp_mat = einsum('ij,jk->ik',tmp_mat,mpo[k][:,:,i_occ[k],j_occ[k]])
            mat[i,j] += tmp_mat[[0]]
    return mat

def createGlobalCurrMPO(hamType,hamParams,conjTrans=False):
    ############################################
    # Determine MPO
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    ham = []
    if hamType == 'tasep':
        # Totally Asymmetric Simple Exclusion Process -----------------------------------------
        alpha = hamParams[0]
        beta = hamParams[1]
        s = hamParams[2]
        exp_alpha = alpha*np.exp(-s)
        exp_beta = beta*np.exp(-s)
        exp_p = np.exp(-s)
        currMPO = [None]*3
        currMPO[0] = np.array([[exp_alpha*Sm,exp_p*Sp,I]])
        currMPO[1] = np.array([[I],[Sm],[exp_beta*Sp]])
        currMPO[2] = np.array([[I ,       z, z],
                               [Sm,       z, z],
                               [z ,exp_p*Sp, I]])
    elif hamType == 'sep':
        # Generic Simple Exclusion Process -----------------------------------------------------
        alpha = float(hamParams[0])
        gamma = float(hamParams[1])
        p = float(hamParams[2])
        q = float(hamParams[3])
        beta = float(hamParams[4])
        delta = float(hamParams[5])
        s = float(hamParams[6])
        exp_alpha = np.exp(-s)*alpha
        exp_beta = np.exp(-s)*beta
        exp_p = np.exp(-s)*p
        exp_q = np.exp(s)*q
        exp_delta = np.exp(s)*delta
        exp_gamma = np.exp(s)*gamma
        currMPO = [None]*3
        currMPO[0] = np.array([[exp_alpha*Sm-exp_gamma*Sp,Sp,Sm,I]])
        currMPO[1] = np.array([[I],[exp_p*Sm],[-exp_q*Sp],[exp_delta*Sp-exp_beta*Sm]])
        currMPO[2] = np.array([[        I,  z,  z, z],
                               [ exp_p*Sm,  z,  z, z],
                               [-exp_q*Sp,  z,  z, z],
                               [        z, Sp, Sm, I]])
    ############################################
    # conjugate transpose operator if desired
    if conjTrans:
        for site in range(len(ham)):
            currMPO[site] = np.transpose(currMPO[site],(0,1,3,2)).conj()
    return currMPO

def createLocalCurrMPO(hamType,hamParams,conjTrans=False):
    ############################################
    # Determine MPO
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    ham = []
    if hamType == 'tasep':
        # Totally Asymmetric Simple Exclusion Process -----------------------------------------
        alpha = hamParams[0]
        beta = hamParams[1]
        s = hamParams[2]
        exp_alpha = alpha*np.exp(-s)
        exp_beta = beta*np.exp(-s)
        exp_p = np.exp(-s)
        currMPO = [None]*2
        currMPO[0] = np.array([[z ,exp_p*Sp, I]])
        currMPO[1] = np.array([[I],[Sm],[z]])
    elif hamType == 'sep':
        # Generic Simple Exclusion Process -----------------------------------------------------
        alpha = float(hamParams[0])
        gamma = float(hamParams[1])
        p = float(hamParams[2])
        q = float(hamParams[3])
        beta = float(hamParams[4])
        delta = float(hamParams[5])
        s = float(hamParams[6])
        exp_alpha = np.exp(-s)*alpha
        exp_beta = np.exp(-s)*beta
        exp_p = np.exp(-s)*p
        exp_q = np.exp(s)*q
        exp_delta = np.exp(s)*delta
        exp_gamma = np.exp(s)*gamma
        currMPO = [None]*2
        currMPO[0] = np.array([[z,Sp,Sm,I]])
        currMPO[1] = np.array([[I],[exp_p*Sm],[-exp_q*Sp],[z]])
    ############################################
    # conjugate transpose operator if desired
    if conjTrans:
        for site in range(len(ham)):
            currMPO[site] = np.transpose(currMPO[site],(0,1,3,2)).conj()
    return currMPO

def createLocalDensMPO():
    n = np.array([[0,0],[0,1]])
    return np.array([[n]])
