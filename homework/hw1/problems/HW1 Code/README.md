# CS 189 Homework 1

## Name: Zhihao Xu
## Student ID: 3034288950

Almost all the codes can be simply executed by clicking run in jupyter notebook. 
Here are some remarks of my program:
1. All the path are relative path. The user need to put the code file in the same file with the data file. Or change it to the absolute path.
2. The executing time of CIFAR-10 may take several minutes, which I think is not inordinate
3. When we apply the model to the testing data, if we want to use the whole training dataset, it may take inordinate amount of time and memory. So, here I just use part of the training data (larger than the training data used in validation part)
4. The output file does not have header. The user should add header "Id" and "Category" manually