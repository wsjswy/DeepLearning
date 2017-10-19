import  numpy as np
import  matplotlib.pyplot as plt

def load_planar_dataset():
    t = np.random.seed(1)

    #样本数量
    m = 400
    N = int(m / 2) # 每类样本的数量

    D = 2  # 维度数
    X = np.zeros((m, D))  # 初始化X
    Y = np.zeros((m, 1))  # 初始化Y
    a = 4  # 花儿的最大长度
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return  X.T, Y.T


def paint_flower():

    X, Y = load_planar_dataset()
    plt.scatter(X[0, :], X[1, :], c=Y, s=40)
    plt.show()

if __name__ == '__main__':
    paint_flower()
