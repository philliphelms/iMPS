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
        alpha = hamParams[0]
        delta = hamParams[1]
        gamma = hamParams[2]
        beta = hamParams[3]
        p = hamParams[4]
        q = hamParams[5]
        s = hamParams[6]
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
        occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(2-len(bin(i)[2:]))+bin(i)[2:])))
    for i in range(d**N):
        i_occ = occ[i,:]
        for j in range(d**N):
            j_occ = occ[j,:]
            tmp_mat = np.array([[1]])
            for k in range(N):
                tmp_mat = einsum('ij,jk->ik',tmp_mat,mpo[k][:,:,i_occ[k],j_occ[k]])
            mat[i,j] += tmp_mat[[0]]
    return mat
