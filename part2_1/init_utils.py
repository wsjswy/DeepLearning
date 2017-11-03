
import  numpy as np
import  sklearn
import  sklearn.datasets
from  matplotlib import  pyplot as plt


def sigmod(x):
    return  1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def compute_loss(a3, Y):
    """
    Implement the loss function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    loss - value of the loss function
    """
    m = Y.shape[1]  #样本个数

    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)

    return 1 / m * np.sum(logprobs)

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    """

    Z = {}
    A = {}

    A['a0'] = X

    L = len(parameters) // 2

    for i in range(1, L - 1):
        Z['z' + str(i)] = np.dot(parameters['W' + str(i)], A['a' + str(i - 1)]) + parameters['b' + str(i)]
        A['a' + str(i)] = relu(Z['z' + str(i)])

    #output layder

    Z['z' + str(L)] = np.dot(parameters['W' + str(L)], A['a' + str(L - 1)]) + parameters['b' + str(L)]

    A['a' + str(L)] = sigmod(Z['z' + str(L)])

    # configure zi, ai, Wi, bi
    cache = {}
    for i in range(1, L):
        cache['z' + str(i)] = Z['z' + str(i)]
        cache['a' + str(i)] = A['a' + str(i)]
        cache['W' + str(i)] = parameters['W' + str(i)]
        cache['b' + str(i)] = parameters['b' + str(i)]

    return A['a' + str(L)], cache



def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    gradients = {}

    return  gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of n_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters['W' + str(i)] = ...
                  parameters['b' + str(i)] = ...
    """
    L = len(parameters) // 2 # hidden layer length

    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters

def predict(X, Y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    return p

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()



def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
      Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

      Arguments:
      X -- input data, of shape (2, number of examples)
      Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
      learning_rate -- learning rate for gradient descent
      num_iterations -- number of iterations to run gradient descent
      print_cost -- if True, print the cost every 1000 iterations
      initialization -- flag to choose which initialization to use ("zeros","random" or "he")

      Returns:
      parameters -- parameters learnt by the model
    """

    grads = {}
    cost = []

    m = X.shape[1] #number of examples

    # init parameters
    parameters = {}

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

    return  parameters

def initialize_parameters_zeros(layers_dims):
    """
       Arguments:
       layer_dims -- python array (list) containing the size of each layer.

       Returns:
       parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                       W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                       b1 -- bias vector of shape (layers_dims[1], 1)
                       ...
                       WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                       bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}

    L = len(layers_dims) + 1

    for i in range(1, L):

        parameters['W' + str(i)] = np.zeros(layers_dims[i], layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros(layers_dims[i], 1)

    return  parameters

def initialize_parameters_random(layers_dims):

    parameters = {}

    L = len(layers_dims) + 1

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.rand(layers_dims[i], layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros(layers_dims[i], 1)


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """

    m = Y.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']




if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_X, train_Y, test_X, test_Y = load_dataset()
