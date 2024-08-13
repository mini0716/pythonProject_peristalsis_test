import cv2

from imports_header import *


def dataOpen(path, MAX_SIZE=5000000):
    audio, sr = librosa.load(path)
    audio = audio[:MAX_SIZE] if len(audio) > MAX_SIZE else np.tile(audio, (MAX_SIZE // len(audio)) + 1)[:MAX_SIZE]

    return audio, sr


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


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
    n_fft, hop_length, n_mels = 2048, 512, 128
    fmin, fmax = 40, 11000
    power, ref, top_db = 2.0, np.std, 255

    S = librosa.feature.melspectrogram(y=X, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=power)        # Spectrogram 생성
    S_db = np.abs(librosa.power_to_db(S, ref=ref, top_db=top_db))                                       # dB 스케일로 변환

    return S_db


def onPreprocessing(X, sr):
    X[0], X[1] = __max__, __min__

    X = MinMaxScaler((-1, 1)).fit_transform(X.reshape(-1, 1)).reshape(-1)

    spectrogram = getSpectrogram(X, sr)

    spectrogram = np.hstack([spectrogram, spectrogram])
    spectrogram = spectrogram[:, :1024]
    # print(spectrogram.shape)

    x = spectrogram

    spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)
    spectrogram = 255 - spectrogram

    spectrogram = np.var([spectrogram, x], axis=0)

    z = 128  # int(spectrogram.shape[1] * 0.15)
    E = (spectrogram.shape[1] // z) * z

    spectrogram = spectrogram[:, :E]
    spectrogram = spectrogram.reshape(128, z, -1)
    # spectrogram = spectrogram.transpose((0, 2, 1))

    # spectrogram = np.sum(spectrogram, axis=2)
    spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)

    spectrogram = spectrogram.transpose((2, 0, 1))
    spectrogram = np.expand_dims(spectrogram, axis=-1)

    # for s in range(10):
    #     cv2.imshow('img', spectrogram[:, :, s])
    #     cv2.waitKey(0)

    return spectrogram


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver4.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver4.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)
    dataSet = [['trainSet', X_train, y_train], ['testSet', X_test, y_test]]

    __max__, __min__ = 1.8101999759674072, -1.7044665813446045

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX0 = data_group.create_dataset('X0', shape=(0, 8, 128, 128, 1), maxshape=(None, 8, 128, 128, 1), dtype=np.uint8)
        labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]), dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)

            X0 = onPreprocessing(audio, sr=32000)

            X0 = X0.reshape((1,) + X0.shape)
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
