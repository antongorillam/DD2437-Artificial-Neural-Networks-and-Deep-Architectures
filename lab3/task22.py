import numpy as np

def getW():

    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]

    w1 = np.outer(x1,x1)
    np.fill_diagonal(w1,0)
    w2 = np.outer(x2,x2)
    np.fill_diagonal(w2,0)
    w3 = np.outer(x3,x3)
    np.fill_diagonal(w3,0)

    W = w1 + w2 + w3
    #print(W)

    #print('x1: ', np.sign(W @ x1) == x1)
    #print('x2: ', x2 == np.sign(W @ x2))
    #print('x3: ', x3 == np.sign(W @ x3))
    return W