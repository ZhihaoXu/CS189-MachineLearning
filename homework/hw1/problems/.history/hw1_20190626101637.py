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


