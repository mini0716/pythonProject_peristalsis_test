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
    X1 = np.interp(np.linspace(0, 1, 2500 * n), np.linspace(0, 1, len(X)), X.reshape(-1))
    X1 = X1.reshape(-1, 1)

    X = X.reshape(n, -1)
    resampled_waveform = np.array([np.interp(np.linspace(0, 1, 2500), np.linspace(0, 1, len(x)), x.reshape(-1)) for x in X])
    resampled_waveform = resampled_waveform.reshape(resampled_waveform.shape)

    return X1, resampled_waveform


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def dataOpen(path):
    __MAX_SIZE__ = 5000000

    audioData, sr = librosa.load(path)

    if len(audioData) > __MAX_SIZE__:
        audiolp = audioData[:__MAX_SIZE__]
    else:
        K = (__MAX_SIZE__ // len(audioData)) + 1
        audiolp = np.tile(audioData, K)[:__MAX_SIZE__]

    return audiolp, sr


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"파일 {file_path} 삭제 성공")
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


if __name__ == '__main__':
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver2.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver2.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)

    dataSet = [['trainSet', X_train, y_train], ['testSet', X_test, y_test]]
    N, C = 20, 2500

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX1 = data_group.create_dataset('X1', shape=(0, C*N, 1),
                                           maxshape=(None, C*N, 1),
                                           dtype=np.float32)
        dataX2 = data_group.create_dataset('X2', shape=(0, N, C),
                                           maxshape=(None, N, C),
                                           dtype=np.float32)
        labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]),
                                           maxshape=(None, Y.shape[1]),
                                           dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)
            X1, X2 = onPreprocessing(audio, sr=sr, n=N)

            X1 = X1.reshape((1,) + X1.shape)
            X2 = X2.reshape((1,) + X2.shape)
            y = y.reshape((1,) + y.shape)

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

            print(f'{subset}] X1: {dataX1.shape}, X2: {dataX2.shape}, y: {labels.shape}')
