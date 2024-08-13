from imports_header import *


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


delete_file(r'D:/dataset/ISIC/NPY/ISIC_ver1.h5')
h5pyfile = h5py.File('D:/dataset/ISIC/NPY/ISIC_ver1.h5', 'a')

X_train, y_train = np.load(r'D:/dataset/ISIC/NPY/Xtrain.npy'), np.load(r'D:/dataset/ISIC/NPY/ytrain.npy')
X_test, y_test = np.load(r'D:/dataset/ISIC/NPY/Xtest.npy'), np.load(r'D:/dataset/ISIC/NPY/ytest.npy')
dataSet = [('trainSet', X_train, y_train), ('testSet', X_test, y_test)]

for subset, X, Y in dataSet:
    data_group = h5pyfile.create_group(subset)
    dataX0 = data_group.create_dataset('X', shape=(0, 384, 640, 3), maxshape=(None, 384, 640, 3), dtype=np.uint8)
    labels = data_group.create_dataset('Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]), dtype=np.uint8)

    for x, y in zip(X, Y):
        x = x.reshape((1,) + x.shape)
        y = y.reshape((1,) + y.shape)

        current_data_size = dataX0.shape[0]
        new_data_size = current_data_size + x.shape[0]
        dataX0.resize(new_data_size, axis=0)
        dataX0[current_data_size:new_data_size, :] = x

        current_data_size = labels.shape[0]
        new_data_size = current_data_size + y.shape[0]
        labels.resize(new_data_size, axis=0)
        labels[current_data_size:new_data_size, :] = y

        print(f'{subset}] X: {dataX0.shape}, y: {labels.shape}')

