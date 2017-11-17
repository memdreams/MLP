import numpy as np
from scipy import io
import os
from pylab import scatter, show, legend, xlabel, ylabel
import matplotlib.pyplot as plt



def splitData(rawData):
    class1 = rawData.get('class_1').T  # label set to 0
    class2 = rawData.get('class_2').T  # label set to 1
    class1 = np.hstack((class1, np.zeros((int(class1.shape[0]), 1))))
    class2 = np.hstack((class2, np.ones((class2.shape[0] // 1, 1))))
    data = np.vstack((class1, class2))
    np.random.shuffle(data)

    train_num = int(0.8 * data.shape[0])
    test_num = int(0.2 * data.shape[0])
    # Combine class1 data and class2 data
    train_data = data[0:train_num]
    test_data = data[-test_num:]
    x_train = train_data.T[0:-1].T
    y_train = train_data.T[-1:].T
    x_test = test_data.T[0:-1].T  # np.vstack((class1[-test_num:], class2[-test_num:]))
    y_test = test_data.T[-1:].T #np.vtack((np.zeros((test_num, 1)), np.ones((test_num, 1))))

    # Min-Max Normalization
    x_train -= x_train.min()
    x_train /= x_train.max()
    x_test -= x_test.min()
    x_test /= x_test.max()

    return x_train, y_train, x_test, y_test

def Sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def Hypothesis(X, theta, activeF='Sigmoid'):
    if activeF is'ReLU':
        h = np.dot(X, theta)
        compareZero = np.zeros((h.shape))
        h = np.maximum(compareZero, h)  #ReLU function
    elif activeF is'Sigmoid':
        h = Sigmoid(np.dot(X, theta))
    return h

def CostFunction(m, theta, x_train, y_train):
    h = Hypothesis(x_train, theta, activeF='Sigmoid')
    J = (1.0/m) * np.sum(-y_train*(np.log(h)) - (1-y_train)*(np.log(1-h)))
    return J


def GradientDescent(m, theta, learning_rate, x_train, y_train):
    theta_zeroed = theta
    # theta_zeroed[0, :] = 0
    grads = np.zeros((x_train.shape[1], 1))
    h = Hypothesis(x_train, theta, activeF='Sigmoid')
    for i in range(x_train.shape[1]):
        loss = h - y_train
        grads[i] = (1.0/m) * np.dot(x_train[:, i], loss)
    theta = theta - learning_rate * grads

    # theta = theta/theta.max()
    return theta


def predict(theta, X, activeF):
    # test new data
    m, n = X.shape
    p = np.zeros(shape=(m, 1))
    h = Hypothesis(X, theta, activeF)
    p = 1 * (h >= 0.5)
    return p

def errorRate(y_estimate, y_real):
    error = 0
    for i in range(y_estimate.shape[0]):
        error += np.abs(y_estimate[i] - y_real[i])
    error_rate = error/y_estimate.shape[0]
    accurate_rate = (1 - error_rate) * 100
    print("Error Rate: ", error_rate)
    print("Right Rate: ", accurate_rate)
    return error

# draw 2D picture of 2 classes
def drawClasses(x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    X = np.hstack((x, y))
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('class 1')
    ylabel('class 2')
    legend(['Class 1', 'Class 2'])
    show()

# mat = io.loadmat("./observed/classify_d99_k60_saved1.mat")

iteration = 20000
learning_rate = 0.5
activeF = 'Sigmoid'  # for small data, linear or non-linear are similar.

datadir = "./observed/"
# Read data from the mat file
for filenames in os.walk(datadir):
    for filename in filenames[-1]:
        print(filename)
        readData = io.loadmat(datadir + filename)
        x_train, y_train, x_test, y_test = splitData(readData)
        m, n = x_train.shape
        # Define theta
        # theta = np.zeros((x_train.shape[1], 1))
        theta = 0.1 * np.random.rand(n, 1)
        j = 1
        J = np.zeros((iteration // 1000 + 1, 1))
        J[0] = CostFunction(m, theta, x_train, y_train)
        for i in range(iteration):
            theta = GradientDescent(m, theta, learning_rate, x_train, y_train)
            if i % 1000 == 0:
                J[j] = CostFunction(m, theta, x_train, y_train)
                if J[j] < 0.001:
                    break
                # update learning rate based on cost function
                if (J[j] - J[j - 1]) > 0:
                    learning_rate *= 0.5
                elif (J[j - 1] - J[j]) < 0.0001:
                    learning_rate *= 2
                j += 1

                # print("updated theta: ", theta)
        print('The Loss is ', J[-2])

        p = predict(theta, x_test, activeF)
        error = errorRate(p, y_test)
        # print(error)

        # drawClasses(x_test, y_test)
        # for_scatter_x = range(0, y_train.shape[0], 1)
        # for_scatter_x1 = range(0, y_test.shape[0], 1)
        #
        # plt.subplot(2, 1, 1)
        # plt.title('y')
        # plt.scatter(for_scatter_x1, y_test.reshape(1, -1))
        # plt.subplot(2, 1, 2)
        # plt.title('My prediction')
        # # pos = (y==1).nonzero()[:1]
        # # neg = (y==0).nonzero()[:1]
        # # plt.plot(x[pos, 4].T, x[pos, 5].T, 'k+', markeredgewidth=2, markersize = 7)
        # # plt.plot(x[neg, 4].T, x[neg, 5].T, 'ko', markeredgewidth='r', markersize=7)
        # plt.scatter(for_scatter_x1, p)
        # plt.show()
        print('Accuracy: %f' % ((y_test[np.where(p == y_test)].size / float(y_test.size)) * 100.0))



# print(mat.keys())


