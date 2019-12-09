# Question 2 Data Partitioning
import scipy.io
import pandas as pd
import numpy as np
import random

mnistData = scipy.io.loadmat('data/mnist_data.mat')  # 读取mat文件

mnistData_X = mnistData["training_data"]
mnistData_y = mnistData["training_labels"]

index = random.sample(range(mnistData_X.shape[0]),10000)

mnistData_X_train = mnistData_X[index]
mnistData_X_validate = np.delete(mnistData_X, index, axis=0)
mnistData_y_train = mnistData_y[index]
mnistData_y_validate = np.delete(mnistData_y, index, axis=0)

print(mnistData_X_train.shape)
print(mnistData_X_validate.shape)
print(mnistData_y_train.shape)
print(mnistData_y_validate.shape)


spamData = scipy.io.loadmat('data/spam_data.mat')  # 读取mat文件
print(spamData.keys())
spamData_X = spamData["training_data"]
spamData_y = spamData["training_labels"]

index = random.sample(range(spamData_X.shape[0]),int(spamData_X.shape[0]*0.2))

spamData_X_train = spamData_X[index]
spamData_X_validate = np.delete(spamData_X, index, axis=0)
spamData_y_train = spamData_y[index]
spamData_y_validate = np.delete(spamData_y, index, axis=0)

print(spamData_X_train.shape)
print(spamData_X_validate.shape)
print(spamData_y_train.shape)
print(spamData_y_validate.shape)


cifar10Data = scipy.io.loadmat('data/cifar10_data.mat')  # 读取mat文件
print(cifar10Data.keys())
cifar10Data_X = cifar10Data["training_data"]
cifar10Data_y = cifar10Data["training_labels"]

index = random.sample(range(cifar10Data_X.shape[0]),5000)

cifar10Data_X_train = cifar10Data_X[index]
cifar10Data_X_validate = np.delete(cifar10Data_X, index, axis=0)
cifar10Data_y_train = cifar10Data_y[index]
cifar10Data_y_validate = np.delete(cifar10Data_y, index, axis=0)

print(cifar10Data_X_train.shape)
print(cifar10Data_X_validate.shape)
print(cifar10Data_y_train)
print(cifar10Data_y_validate)

from sklearn import svm
from sklearn.metrics import accuracy_score

data = mnistData
data_X = data["training_data"]
data_y = data["training_labels"]

index = random.sample(range(data_X.shape[0]),10000)

data_X_train = data_X[index]
data_X_validate = np.delete(data_X, index, axis=0)
data_y_train = data_y[index]
data_y_validate = np.delete(data_y, index, axis=0)

print(data_X_train.shape)
print(data_X_validate.shape)
print(data_y_train.shape)
print(data_y_validate.shape)


def svmLinear(data,training_sample):
    data_X = data["training_data"]
    data_y = data["training_labels"]

    index = random.sample(range(data_X.shape[0]),training_sample)

    data_X_train = data_X[index]
    data_X_validate = np.delete(data_X, index, axis=0)
    data_y_train = data_y[index]
    data_y_validate = np.delete(data_y, index, axis=0)

    classifier=svm.SVC(C=1,kernel='linear',max_iter=100)
    classifier.fit(data_X_train,data_y_train.ravel())
    y0 = classifier.predict(data_X_validate)
    print(accuracy_score(y0,data_y_validate))

svmLinear(mnistData,100)
