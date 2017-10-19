import  numpy as np
import  sklearn.datasets
import  sklearn
import  sklearn.linear_model as l_model
import  matplotlib.pyplot as plt

import  part1_3.MyDataSet as MyDataSet


def use_linearmodel(X, Y):

    clf = l_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    plot_decision_boundry(lambda x: clf.predict(x), X, Y)
    plt.title('Logistic Regresstion')
    #LR_predictions = clf.predict(X.T)
    LR_predictions = clf.predict(X.T)

    print('predict_matrix_shape: ' + str(LR_predictions.shape))
    print('Y_matrix_shape: ' + str(Y.shape))

    rightPredictions = np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)

    print(rightPredictions / Y.shape[1])
    plt.show()



def plot_decision_boundry(model, X, y):
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


if __name__ == '__main__':
    X, Y = MyDataSet.load_planar_dataset()

    use_linearmodel(X, Y)