import  numpy as np

def L1Func(yhat, y):

    return  np.sum(abs(yhat - y))


def L2Func(yhat, y):

    return  np.sum(np.power((y - yhat), 2))



if __name__ == '__main__':

    print('test--func')