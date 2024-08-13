from imports_header import *













def dataLoad(path=r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver4.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['X1'], trainSet['X2'], trainSet['Y'][:], \
            testSet['X0'],  testSet['X1'],  testSet['X2'],  testSet['Y'][:]


if __name__ == '__main__':
    trainX0, trainX1, trainX2, trainY, testX0, testX1, testX2, testY = dataLoad()
    print(trainX0.shape, trainX1.shape, trainX2.shape, trainY.shape)
    print(testX0.shape, testX1.shape, testX2.shape, testY.shape)
