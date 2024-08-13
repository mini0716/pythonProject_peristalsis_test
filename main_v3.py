from imports_header import *


def joinPath(path):
    return glob.glob(path + "/**/*/**/*", recursive=True)


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
    # return librosa.power_to_db(librosa.feature.melspectrogram(y=X, sr=sr, n_mels=40), ref=np.max)


def apply_moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def moving_average(data, window_size):
    window_size = window_size + 1 if window_size % 2 == 0 else window_size

    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    return smoothed_data


def median_filter(data, window_size):
    window_size = window_size + 1 if window_size % 2 == 0 else window_size

    half_window = window_size // 2
    smoothed_data = np.zeros_like(data)

    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window : i + half_window + 1]
        smoothed_data[i] = np.median(window)

    return smoothed_data


def onPreprocessing(X, sr):
    spectrogram = getSpectrogram(X, sr)
    spectrogram = np.round(MinMaxScaler().fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255,
                           0).astype(np.uint8)

    spectrogram = spectrogram[:-1, :9216]
    spectrogram = spectrogram.reshape(1024, 3, -1).transpose((1, 0, 2)).reshape(1024*3, -1)
    spectrogram = cv2.resize(spectrogram, (512, 512))

    return spectrogram, 0, 0


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def dataOpen(path, MAX_SIZE=5000000):
    audio, sr = librosa.load(path)
    audio = audio[:MAX_SIZE] if len(audio) > MAX_SIZE else np.tile(audio, (MAX_SIZE // len(audio)) + 1)[:MAX_SIZE]

    return audio, sr


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


if __name__ == '__main__':
# def myDataSet():
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver3.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver3.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)
    dataSet = [['trainSet', X_train, y_train], ['testSet', X_test, y_test]]

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX0 = data_group.create_dataset('X0', shape=(0, 512, 512, 1), maxshape=(None, 512, 512, 1),
                                           dtype=np.uint8)
        labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]),
                                           dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)
            X0, X1, X2 = onPreprocessing(audio, sr=sr)

            X0 = X0.reshape((1,) + X0.shape + (1,))
            y = y.reshape((1,) + y.shape)

            current_data_size = dataX0.shape[0]
            new_data_size = current_data_size + X0.shape[0]
            dataX0.resize(new_data_size, axis=0)
            dataX0[current_data_size:new_data_size, :] = X0

            current_data_size = labels.shape[0]
            new_data_size = current_data_size + y.shape[0]
            labels.resize(new_data_size, axis=0)
            labels[current_data_size:new_data_size, :] = y

            print(f'{subset}] X: {dataX0.shape}, y: {labels.shape}')
