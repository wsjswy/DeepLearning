import  numpy as np
import  matplotlib.pyplot as plt

import  part1_3.basic_sigmod as basic_sigmod
import  part1_3.MyDataSet as MyDataSet
import  part1_3.MyLogistic as MyLogistic

def layer_size(X, y):

    n_x = X.shape[0] #样本特征数
    n_h = 4 # hidden_layer
    n_y = y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2)  #保证每次生成的随机数相同

    W1 = np.random.rand(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return  parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    a1 = np.tanh(Z1)
    Z2 = np.dot(W2, a1) + b2
    a2 = basic_sigmod.sigmod_func(Z2)

    assert(a2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": a1,
             "Z2": Z2,
             "A2": a2}

    return  a2, cache


def compute_cost(a2, Y, parameters): #损失函数

    m = Y.shape[1] #样本数量
    logprobs = np.multiply(np.log(a2), Y) + (1 - Y) * np.multiply(np.log(1 - a2), (1 - Y)) #损失函数
    cost = -(1.0/m)*np.sum(logprobs)
    return  cost


def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1] #样本数量

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 -Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # tanh函数求导结果
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return  grads

def update_parameters(parameters, grads, learning_rate = 0.01):
    new_parameters= {}
    for key in parameters:
        new_parameters[key] = parameters[key] - learning_rate * grads['d' +key]

    return new_parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    np.random.seed(3)

    n_x = X.shape[0]
    n_y = Y.shape[0]


    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations): #更新参数迭代
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 10)

        # if i % 1000 == 0.0:
        #     print("Cost after iteration %i: %f" % (i, cost))

    return  parameters

def predict_point(parameters, X):
    A2, cache = forward_propagation(X, parameters)

    print(A2)
    predictions = A2 > 0.5
    return  predictions

def test_cases():

    X, Y =  MyDataSet.load_planar_dataset()

    new_parameters = nn_model(X, Y, 10)
    # print("W1 = " + str(new_parameters["W1"]))
    # print("b1 = " + str(new_parameters["b1"]))
    # print("W2 = " + str(new_parameters["W2"]))
    # print("b2 = " + str(new_parameters["b2"]))

    t_X = np.array([[ -0.18533515],
            [ 0.15397529]])

    n = predict_point(new_parameters, t_X)
    print(n)

    MyLogistic.plot_decision_boundry(lambda x: predict_point(new_parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

    predictions = predict_point(new_parameters, X)
    print('Accuracy: %d' % float(  \
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    plt.show()

if __name__ == '__main__':
    test_cases()