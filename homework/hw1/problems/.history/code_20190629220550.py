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


spamData = processData("spam_data",0.2)
print(spamData["X_train"].shape)