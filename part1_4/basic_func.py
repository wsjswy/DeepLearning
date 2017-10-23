import  numpy as np


def sigmoid_fun(X):

    A = 1 / (1 + np.exp(-X))

    return  A, X

def sigmoid_backfoward(dA, cache):

    Z = cache

    s, s1 = sigmoid_fun(Z)

    dZ = dA * s * (1 -s)

    return  dZ

def relu_func(X):

    A = np.maximum(0, X)

    assert(A.shape, X.shape)

    return  A, X

def relu_backfowrd(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    assert(dZ.shape, Z.shape)

    return dZ