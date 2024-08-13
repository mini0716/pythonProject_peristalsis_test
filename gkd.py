from imports_header import *


def dataOpen(path, MAX_SIZE=5000000):
    audio, sr = librosa.load(path)
    audio = audio[:MAX_SIZE] if len(audio) > MAX_SIZE else np.tile(audio, (MAX_SIZE // len(audio)) + 1)[:MAX_SIZE]

    return audio, sr


def audioFileLoad(path):
    from tensorflow.keras.utils import to_categorical
    files, labels = [], []

    for label, subNames in enumerate(os.listdir(path)):
        for file in os.listdir(f'{path}/{subNames}'):
            if file.endswith('.ogg'):
                files.append(os.path.join(path, subNames, file))
                labels.append(label)

    return np.array(files), to_categorical(labels)


def getSpectrogram(X, sr):
    return librosa.amplitude_to_db(np.abs(librosa.stft(X)), ref=np.max)


def onPreprocessing(X, sr):
    X = MinMaxScaler().fit_transform(X.reshape(-1, 1)).reshape(-1)
    X = getSpectrogram(X, sr)

    X0, X2 = X[10:, 10:], X[:-10, :-10]
    spectrogram = X1 = X[5:-5, 5:-5]

    X = np.var([X0, X1, X2], axis=0)
    X = np.round(MinMaxScaler().fit_transform(X.reshape(-1, 1)).reshape(X.shape) * 255, 0).astype(np.uint8)
    spectrogram = np.round(MinMaxScaler().fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)
    X = cv2.equalizeHist(X)

    X = np.stack([spectrogram, X, X], axis=-1)
    X = cv2.resize(X, (X.shape[1]//4, 256))

    X = X[:, :X.shape[1] // X.shape[0] * X.shape[0], :]
    X = X.reshape(X.shape[0], -1, X.shape[0], 3).transpose((1, 0, 2, 3))

    X = np.round(MinMaxScaler().fit_transform(X.reshape(-1, 1)).reshape(X.shape) * 255, 0).astype(np.uint8)

    xfd = np.sum(X, axis=(1, 2, 3))
    xsort = np.argsort(xfd)

    Xmax = np.array([X[i] for i in np.sort(xsort[:5])])
    Xmin = np.array([X[i] for i in np.sort(xsort[-5:])])
    # print(X.shape, Xmax.shape, Xmin.shape)

    return X, Xmax, Xmin


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver5.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver5.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)
    dataSet = [['trainSet', X_train, y_train],
               [ 'testSet', X_test,  y_test ]]

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX0 = data_group.create_dataset('X0', shape=(0, 9, 256, 256, 3), maxshape=(None, 9, 256, 256, 3),
                                           dtype=np.uint8)
        dataX1 = data_group.create_dataset('X1', shape=(0, 5, 256, 256, 3), maxshape=(None, 5, 256, 256, 3),
                                           dtype=np.uint8)
        dataX2 = data_group.create_dataset('X2', shape=(0, 5, 256, 256, 3), maxshape=(None, 5, 256, 256, 3),
                                           dtype=np.uint8)
        labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]), dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)
            X0, X1, X2 = onPreprocessing(audio, sr=sr)

            X0 = X0.reshape((1,) + X0.shape)
            X1 = X1.reshape((1,) + X1.shape)
            X2 = X2.reshape((1,) + X2.shape)
            y = y.reshape((1,) + y.shape)

            current_data_size = dataX0.shape[0]
            new_data_size = current_data_size + X0.shape[0]
            dataX0.resize(new_data_size, axis=0)
            dataX0[current_data_size:new_data_size, :] = X0

            current_data_size = dataX1.shape[0]
            new_data_size = current_data_size + X1.shape[0]
            dataX1.resize(new_data_size, axis=0)
            dataX1[current_data_size:new_data_size, :] = X1

            current_data_size = dataX2.shape[0]
            new_data_size = current_data_size + X2.shape[0]
            dataX2.resize(new_data_size, axis=0)
            dataX2[current_data_size:new_data_size, :] = X2

            current_data_size = labels.shape[0]
            new_data_size = current_data_size + y.shape[0]
            labels.resize(new_data_size, axis=0)
            labels[current_data_size:new_data_size, :] = y

            print(f'{subset}] X: {dataX0.shape}, {dataX1.shape}, {dataX2.shape}, y: {labels.shape}')
