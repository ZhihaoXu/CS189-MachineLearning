import scipy.io
import pandas as pd
import numpy as np
import random

def processData(name,testing_size):
    path = 'data/' + name + ".mat" 
    data = scipy.io.loadmat(path)  # 读取mat文件

    data_X = data["training_data"]
    data_y = data["training_labels"]
    data_t = data["test_data"]
    
    if testing_size <= 1:
        testing_size = int(testing_size * data_X.shape[0])
    
    # random.seed(189)
    index = random.sample(range(data_X.shape[0]),data_X.shape[0]-testing_size)

    data_X_train = data_X[index]
    data_X_validate = np.delete(data_X, index, axis=0)
    data_y_train = data_y[index]
    data_y_validate = np.delete(data_y, index, axis=0)

    Data = dict()
    Data["X_train"] = data_X_train
    Data["X_validate"] = data_X_validate
    Data["y_train"] = data_y_train
    Data["y_validate"] = data_y_validate
    Data["test"] = data_t
    return Data

mnistData = processData("mnist_data",10000)
spamData = processData("spam_data",0.2)
print(spamData["X_train"].shape)
cifar10Data = processData("cifar10_data",5000)
print(cifar10Data["X_train"].shape)

from sklearn import svm
from sklearn.metrics import accuracy_score

def svmFit(data,training_sample,c=1,kernel="linear",gam="scale"):
    data_X = data["X_train"]
    data_y = data["y_train"]
    
    # random.seed(189)
    index = random.sample(range(data_X.shape[0]),training_sample)

    data_X_train = data_X[index]
    data_X_validate = data["X_validate"]
    data_y_train = data_y[index]
    data_y_validate = data["y_validate"]

    classifier=svm.SVC(C=c,kernel=kernel,max_iter=-1,gamma=gam)
    classifier.fit(data_X_train,data_y_train.ravel())
    
    y_validate = classifier.predict(data_X_validate)
    validate_accuracy = accuracy_score(y_validate,data_y_validate)
    
    y_train = classifier.predict(data_X_train)
    train_accuracy = accuracy_score(y_train,data_y_train)
    return train_accuracy,validate_accuracy