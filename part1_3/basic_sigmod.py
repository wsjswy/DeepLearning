import  math


import  numpy as np
def sigmod_func(z):

   # return  1 / (1 + math.exp(0 - z))

    return  1 / (1 + np.exp(0 - z))

def sigmod_gradizet(z):

    sigmod_value = sigmod_func(z)

    return  sigmod_value * (1 - sigmod_value)


def image2vector(image):

    a, b, c = image.shape

    v = image.reshape(a * b * c, 1)

    return  v


def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims = True)  #计算每一行的长度，得到一个列向量

    print(x_norm)

    y_norm = np.linalg.norm(x, axis=0, keepdims= True)

    print(y_norm)

    x = x / x_norm

    return  x


def softmax(x):
    x_exp  = np.exp(x)

    print(x_exp)

    x_sum = np.sum(x, axis = 1, keepdims = True)

    s = x_exp / x_sum

    return  s


def test_softmax():

    x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0],
                  [7, 5, 0, 0, 0]])

    print(softmax(x))

def test_image2vector():
    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])

    a = image2vector(image)

    print(a)

if __name__ == '__main__':
    test_softmax()
