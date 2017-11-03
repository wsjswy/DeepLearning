import  numpy as np

import  part2_2.helpFunc as helpfunc
import  part2_2.gradientdescent as gd


def  mini_batch(layers, datainput, labels, number_iterations):

    parameters = helpfunc.initialize_parameters(layers)


    for i in range(number_iterations):
        a, cache = helpfunc.forward_propagation(datainput, parameters)

        #comput cost

        cost = helpfunc.compute_cost(a, labels)

        #backfoward

        grads = helpfunc.backward_propagation(a, cache, parameters)

        #update parameters

        parameters = gd.update_parameters_with_gd(parameters, grads)

def random_mini_batches_test_case():
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    return X, Y, mini_batch_size


if __name__ == '__main__':


    layer_dims = [3, 4, 3,1]

