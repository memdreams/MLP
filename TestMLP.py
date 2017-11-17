import numpy as np
import random
from scipy import io
import os
from MLPModel import NeuralNetwork
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

def errorRate(y_estimate, y_real):
    error = 0
    for i in range(y_estimate.shape[0]):
        error += np.abs(y_estimate[i] - y_real[i])
    error_rate = error/y_estimate.shape[0]
    accurate_rate = (1 - error_rate) * 100
    print("Error Rate: ", error_rate)
    print("Right Rate: ", accurate_rate)
    return error


def splitData(rawData):
    class1 = rawData.get('class_1').T  # label set to 0
    class2 = rawData.get('class_2').T  # label set to 1
    class1 = np.hstack((class1, np.zeros((int(class1.shape[0]), 1))))
    class2 = np.hstack((class2, np.ones((class2.shape[0]//1, 1))))
    data = np.vstack((class1, class2))
    np.random.shuffle(data)

    train_num = int(0.8 * data.shape[0])
    test_num = int(0.2 * data.shape[0])
    # Combine class1 data and class2 data
    train_data = data[0:train_num]
    test_data = data[-test_num:]
    x_train = train_data.T[0:-1].T
    y_train = train_data.T[-1:].T
    x_test = test_data.T[0:-1].T #np.vstack((class1[-test_num:], class2[-test_num:]))
    y_test = test_data.T[-1:].T #np.vstack((np.zeros((test_num, 1)), np.ones((test_num, 1))))

    # # Min-Max Normalization
    # x_train -= x_train.min()
    # x_train /= x_train.max()
    # x_test -= x_test.min()
    # x_test /= x_test.max()
    return x_train, y_train, x_test, y_test

# mat = io.loadmat("./observed/classify_d3_k2_saved1.mat")
# X_train, y_train, X_test, y_test = splitData(mat)
# m, n = X_train.shape


# nn = NeuralNetwork([n, 20, 1], 'tanh')
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# labels_train = LabelBinarizer().fit_transform(y_train)
# labels_test = LabelBinarizer().fit_transform(y_test)
# print("start fitting")


datadir = "./observed/"
# Read data from the mat file
for filenames in os.walk(datadir):
    for filename in filenames[-1]:
        print(filename)
        readData = io.loadmat(datadir + filename)
        X_train, y_train, X_test, y_test = splitData(readData)
        m, n = X_train.shape

        # nn = NeuralNetwork([n, 150, 1], 'sigmoid')  # for 99 features data
        nn = NeuralNetwork([n, 50, 1], 'tanh')  # for small features data

        nn.fit(X_train, y_train, epochs=20000)
        predictions = []
        for i in range(X_test.shape[0]):
            o = nn.predic(X_test[i])
            # predictions.append(np.argmax(o))
            if o[0] > 0.5:
                pred = 1
            else:
                pred = 0
            predictions.append(pred)
        # print(predictions)
        predict = np.array(predictions)
        error = errorRate(predict, y_test)

# mat = io.loadmat("./observed/classify_d4_k3_saved2.mat")
# X_train, y_train, X_test, y_test = splitData(mat)
# m, n = X_train.shape
#
#
# nn = NeuralNetwork([n, 35, 1], 'sigmoid')
# nn.fit(X_train, y_train, epochs=20000)
# predictions = []
# for i in range(X_test.shape[0]):
#     o = nn.predic(X_test[i])
#     # predictions.append(np.argmax(o))
#     if o[0] > 0.5:
#         pred = 1
#     else:
#         pred = 0
#     predictions.append(pred)
# # print(predictions)
# predict = np.array(predictions)
# error = errorRate(predict, y_test)
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
