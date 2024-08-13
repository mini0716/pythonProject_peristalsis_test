from imports_header import *


def joinPath(path):
    return glob.glob(path + "/**/*/**/*", recursive=True)


def audioFileLoad(path):
    from tensorflow.keras.utils import to_categorical
    files, labels = [], []

    for label, subNames in enumerate(os.listdir(path)):
        for file in os.listdir(f'{basePath}/{subNames}'):
            if file.endswith('.ogg'):
                files.append(os.path.join(basePath, subNames, file))
                labels.append(label)

    return np.array(files), to_categorical(labels)


def getSpectrogram(X, sr):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=X, sr=sr), ref=np.max)


def onPreprocessing(X, sr, n=20):
    x0, x1 = X[X < 0], X[X > 0]

    x0 = np.interp(np.linspace(0, 1, 224*25), np.linspace(0, 1, len(x0)), x0)
    x1 = np.interp(np.linspace(0, 1, 224*25), np.linspace(0, 1, len(x1)), x1)

    x0 = MinMaxScaler((-1, 1)).fit_transform(x0.reshape(-1, 1)).reshape(-1)
    x1 = MinMaxScaler((-1, 1)).fit_transform(x1.reshape(-1, 1)).reshape(-1)

    imgs = x0.reshape(25, 1, 224) * x1.reshape(25, 224, 1)

    for i in range(2):
        for j in range(2):
            imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112].sort(axis=1)
            imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112].sort(axis=2)

            if i == 0 and j == 1:
                imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112] = imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112][:, ::-1, :]
            if i == 1 and j == 0:
                imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112] = imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112][:, :, ::-1]
            if i == 1 and j == 1:
                imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112] = imgs[:, j * 112:(j + 1) * 112, i * 112:(i + 1) * 112][:, ::-1, ::-1]

    imgs = imgs.reshape(imgs.shape + (1,))

    # print(imgs.shape)

    # spectrogram0 = getSpectrogram(x0, sr)
    # spectrogram1 = getSpectrogram(x1, sr)

    # for img in imgs:
    #     plt.imshow(img)
    #     plt.show()


    return imgs


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def dataOpen(path):
    audiolp, sr = librosa.load(path)

    # __MAX_SIZE__ = 5000000
    # if len(audioData) > __MAX_SIZE__:
    #     audiolp = audioData[:__MAX_SIZE__]
    # else:
    #     K = (__MAX_SIZE__ // len(audioData)) + 1
    #     audiolp = np.tile(audioData, K)[:__MAX_SIZE__]

    return audiolp, sr


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"파일 {file_path} 삭제 성공")
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


if __name__ == '__main__':
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver1.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver1.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)

    dataSet = [['trainSet', X_train, y_train], ['testSet', X_test, y_test]]
    N, C = 20, 2500

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX1 = data_group.create_dataset('X1', shape=(0, 25, 224, 224, 1),
                                           maxshape=(None, 25, 224, 224, 1),
                                           dtype=np.float32)
        # dataX2 = data_group.create_dataset('X2', shape=(0, N, 224, 224, 1),
        #                                    maxshape=(None, N, 224, 224, 1),
        #                                    dtype=np.float32)
        labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]),
                                           maxshape=(None, Y.shape[1]),
                                           dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)
            X1 = onPreprocessing(audio, sr=sr, n=N)

            X1 = X1.reshape((1,) + X1.shape)
            # X2 = X2.reshape((1,) + X2.shape)
            y = y.reshape((1,) + y.shape)
            #
            current_data_size = dataX1.shape[0]
            new_data_size = current_data_size + X1.shape[0]
            dataX1.resize(new_data_size, axis=0)
            dataX1[current_data_size:new_data_size, :] = X1
            #
            # current_data_size = dataX2.shape[0]
            # new_data_size = current_data_size + X2.shape[0]
            # dataX2.resize(new_data_size, axis=0)
            # dataX2[current_data_size:new_data_size, :] = X2
            #
            current_data_size = labels.shape[0]
            new_data_size = current_data_size + y.shape[0]
            labels.resize(new_data_size, axis=0)
            labels[current_data_size:new_data_size, :] = y
            #
            # print(f'{subset}] X1: {dataX1.shape}, X2: {dataX2.shape}, y: {labels.shape}')
            print(f'{subset}] X1: {dataX1.shape}, y: {labels.shape}')
