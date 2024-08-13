from imports_header import *


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def joinPath(path):
    def list_files_in_directory(directory):
        PATH = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                PATH.append(os.path.join(root, file))

        return PATH

    files = list_files_in_directory(path)
    np.random.shuffle(files)

    return files


def dataOpen(path):
    y = path[path.find('_')+1:]
    y = y[:y.find('_')] if '_' in y else y[:y.find('.')]

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img, y


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


if __name__ == '__main__':
# def myDataSet():
    basePath = 'D://dataset//BirdCLEF 2023//mk//img//trainSet//'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_AUG_ver4.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_AUG_ver4.h5', 'a')

    data_group = h5pyfile.create_group('trainSet')
    dataX0 = data_group.create_dataset('X0', shape=(0, 512, 512, 1), maxshape=(None, 512, 512, 1), dtype=np.uint8)
    labels = data_group.create_dataset('Y', shape=(0, 247), maxshape=(None, 247), dtype=np.float32)

    PATHS = joinPath(basePath)

    for path in PATHS:
        img, y = dataOpen(path)
        img = img.reshape((1,) + img.shape + (1,))
        Y = np.zeros((1, 247))
        Y[:, int(y)] = 1

        current_data_size = dataX0.shape[0]
        new_data_size = current_data_size + img.shape[0]
        dataX0.resize(new_data_size, axis=0)
        dataX0[current_data_size:new_data_size, :] = img

        current_data_size = labels.shape[0]
        new_data_size = current_data_size + Y.shape[0]
        labels.resize(new_data_size, axis=0)
        labels[current_data_size:new_data_size, :] = Y

        print(f'trainSet] X: {dataX0.shape}, y: {labels.shape}')

    ####################################################################################################################
    basePath = 'D://dataset//BirdCLEF 2023//mk//img//testSet//'

    data_group = h5pyfile.create_group('testSet')
    dataX0 = data_group.create_dataset('X0', shape=(0, 512, 512, 1), maxshape=(None, 512, 512, 1), dtype=np.uint8)
    labels = data_group.create_dataset('Y', shape=(0, 247), maxshape=(None, 247), dtype=np.float32)

    PATHS = joinPath(basePath)

    for path in PATHS:
        img, y = dataOpen(path)
        img = img.reshape((1,) + img.shape + (1,))
        Y = np.zeros((1, 247))
        Y[:, int(y)] = 1

        current_data_size = dataX0.shape[0]
        new_data_size = current_data_size + img.shape[0]
        dataX0.resize(new_data_size, axis=0)
        dataX0[current_data_size:new_data_size, :] = img

        current_data_size = labels.shape[0]
        new_data_size = current_data_size + Y.shape[0]
        labels.resize(new_data_size, axis=0)
        labels[current_data_size:new_data_size, :] = Y

        print(f'testSet] X: {dataX0.shape}, y: {labels.shape}')
