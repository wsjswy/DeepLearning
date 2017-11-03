import part2_2.helpFunc as helpfunc

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2


    for i in range(0, L):
        parameters['W' + str(i + 1)] = parameters['W' + str(i)] - grads['dW' + str(i + 1)] * learning_rate
        parameters['b' + str(i + 1)] = parameters['W' + str(i)] - grads['dW' + str(i + 1)] * learning_rate

    return  parameters


def batch_gradient_descent(input_data, labels, layer_dims, num_iterations, learning_rate):

    parameters = helpfunc.initialize_parameters(layer_dims)

    for i in range(num_iterations):

        #正向传播
        a, caches  = helpfunc.forward_propagation(input_data, parameters)

        #cost fun

        costs = helpfunc.compute_cost(a, labels)

        # backfoward

        grads = helpfunc.backward_propagation(input_data, caches, parameters)


        #update parameters

        parameters = update_parameters_with_gd(parameters, grads, learning_rate)

    return parameters

def stochastic_gradient_descent(input_data, labels, layer_dims, num_iterations, m, learning_rate):

    parameters = helpfunc.initialize_parameters(layer_dims)

    for i in range(num_iterations):
        for j in range(0, m):

            # forward
            a, caches = helpfunc.forward_propagation(input_data[:,j], parameters)

            # cost fun
            cost = helpfunc.compute_cost(a, labels[:,j])

            #backfoward

            grads = helpfunc.backward_propagation(input_data, caches, parameters)

            #update parameters

            parameters = update_parameters_with_gd(parameters, grads, learning_rate)


    return  parameters