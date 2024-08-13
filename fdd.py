import random

import numpy as np

from imports_header import *


X, y = np.load(f"D:/dataset/ISIC/NPY/datas.npy"), np.load(f"D:/dataset/ISIC/NPY/labels.npy")

c = list(zip(X, y))
random.shuffle(c)
X, y = zip(*c)
X, y = np.array(X), np.array(y)
y = to_categorical(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

np.save(r'D:/dataset/ISIC/NPY/Xtrain.npy', X_train), np.save(r'D:/dataset/ISIC/NPY/ytrain.npy', y_train)
np.save(r'D:/dataset/ISIC/NPY/Xtest.npy', X_test), np.save(r'D:/dataset/ISIC/NPY/ytest.npy', y_test)
