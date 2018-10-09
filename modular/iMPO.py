import numpy as np

def getHam(hamType,params):
    # Important Operators
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    if hamType == "tasep":
        # Collect Parameters
        a = params[0]
        b = params[1]
        s = params[2]
        p = 1.
        # Weight with bias
        exp_a = np.exp(-s)*alpha
        exp_b = np.exp(-s)*beta
        exp_p = np.exp(-s)*p
        # 

       

