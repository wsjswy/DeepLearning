import  numpy as np


import part1_4.basic_func  as basicFunc

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.rand(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    #是否检查矩阵的格式

    #装载参数
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }

    return  parameters

def initialize_parameters_deep(layer_dims):

    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    L = len(layer_dims)
    parameters = {}
    for  i in range(1, L):
        parameters['W' + str(i)] = np.random.rand(layer_dims[i], layer_dims[i -1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return  parameters


def linear_forward(A, W, b):

    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return  Z, cache

def linear_actions_forward(A_prev, W, b, action):
    """
      Implement the forward propagation for the LINEAR->ACTIVATION layer

      Arguments:
      A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
      W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
      b -- bias vector, numpy array of shape (size of the current layer, 1)
      activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

      Returns:
      A -- the output of the activation function, also called the post-activation value
      cache -- a python dictionary containing "linear_cache" and "activation_cache";
               stored for computing the backward pass efficiently
      """

    Z, linear_cache = linear_forward(A_prev, W, b)
    if action == 'sigmoid':
        A, action_cache = basicFunc.sigmoid_fun(Z)
    elif action == 'relu':
        A, action_cache = basicFunc.relu_func(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, action_cache)

    return  A, cache


def L_model_fowrad(X, parameters):
    caches = []
    A = X
    L = len(parameters)

    for i in range(1, L):  #隐藏层，使用relu函数
        A_prev = A
        A, cache = linear_actions_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], 'relu')
        caches.append(cache)

    AL,cache = linear_actions_forward(A, parameters['W' + str(L)],parameters['b' + str(L)], 'sigmoid')

    caches.append(cache) # L层神经网络的缓存列表

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y): #代价函数
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m  # fix me
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    """
     Implement the linear portion of backward propagation for a single layer (layer l)

     Arguments:
     dZ -- Gradient of the cost with respect to the linear output (of current layer l)
     cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

     Returns:
     dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
     dW -- Gradient of the cost with respect to W (current layer l), same shape as W
     db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev) / m  # fix me
    dA_prev = np.dot(W.T, dZ)
    db = np.sum(dZ, axis=1, keepdims=True) / m

    return  dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
     Implement the backward propagation for the LINEAR->ACTIVATION layer.

     Arguments:
     dA -- post-activation gradient for current layer l
     cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
     activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

     Returns:
     dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
     dW -- Gradient of the cost with respect to W (current layer l), same shape as W
     db -- Gradient of the cost with respect to b (current layer l), same shape as b
     """

    if(check_activation(activation)):
        return

    linear_cache, activation_cache = cache
    dZ = None

    if activation == 'rule':
        dZ = basicFunc.relu_backfowrd(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ  = basicFunc.sigmoid_backfoward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return  dA_prev, dW, db

def check_activation(activation):

    activations = []

    activations.append('relu')
    activations.append('sigmoid')

    if activation not in activations:
        return  True
    else:
        return  False

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads  = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network fix me

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        ### END CODE HERE ###
    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    costs = []
    grads = {}




    n_x = X.shape[1]
    n_h = 4  #
    n_y = Y.shape[1]

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #梯度迭代
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_actions_forward(X, W1, b1, action="relu")
        A2, cache2 = linear_actions_forward(A1, W2, b2, action="sigmoid")
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        ### END CODE HERE ###

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

    return  parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    parameters = initialize_parameters_deep(layers_dims) #此时layers_dims 存储了神经网络所有层次的节点信息


    for i in range(0, num_iterations):

        AL, caches = L_model_fowrad(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)


    return  parameters
