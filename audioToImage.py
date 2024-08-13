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
    # X[0], X[1] = 1.8101999759674072, -1.7044665813446045

    X = MinMaxScaler((-1, 1)).fit_transform(X.reshape(-1, 1)).reshape(-1)

    spectrogram = getSpectrogram(X, sr)

    spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)
    spectrogram = 255 - spectrogram

    # spectrogram = spectrogram[:, :9765]

    spectrogram = spectrogram.reshape(128, -1, 2)
    spectrogram = np.sum(spectrogram, axis=-1)
    spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)

    spectrogram = spectrogram[:, :4880]
    spectrogram = spectrogram.reshape(128, 5, -1)
    spectrogram = spectrogram.transpose((1, 0, 2))

    spectrogram = np.vstack(spectrogram)

    spectrograms = []

    for wize in range(0, spectrogram.shape[1]-spectrogram.shape[0]+1, 10):
        spectrograms.append(spectrogram[:, wize:wize+spectrogram.shape[0]])

    spectrogram = np.array(spectrograms)
    # print(spectrogram.shape)

    # spectrogram = np.sum(spectrogram, axis=1)
    # spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)

    # s1 = spectrogram[:, :, 0]
    # s2 = spectrogram[:, :, 1]

    # cv2.imshow(f'Pre Mel spectrogram1 {spectrogram.shape}', spectrogram)
    # cv2.imshow(f'Pre Mel spectrogram2 {s2.shape}', s2)


    # E = 8192
    # spectrogram = spectrogram[:, :E]
    # spectrogram = spectrogram.reshape(128, 1024, -1)
    #
    # spectrogram = np.sum(spectrogram, axis=-1)
    # print(spectrogram.shape)
    #
    # spectrogram = np.round(MinMaxScaler((0, 1)).fit_transform(spectrogram.reshape(-1, 1)).reshape(spectrogram.shape) * 255, 0).astype(np.uint8)
    # spectrogram = cv2.equalizeHist(spectrogram)

    # _, mask = cv2.threshold(spectrogram, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # spectrogram[_ * 1.8 > spectrogram] = 0

    # cv2.imshow(f'Pre Mel spectrogram {spectrogram.shape}', spectrogram)
    # cv2.waitKey(0)

    # spectrogram = spectrogram.reshape(128, -1, 128)
    # spectrogram = spectrogram.transpose((1, 0, 2))

    # spectrogram = spectrogram[:, :, :-2]

    # print(spectrogram.shape)
    #
    # for img in spectrogram:
    #     img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #     cv2.imshow(f'Pre Mel spectrogram {img.shape}', img)
    #     # cv2.imshow(f'Pre Mel laplacian  {unsharp_mask .shape}', unsharp_mask)
    #     cv2.waitKey(0)

    return spectrogram


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    num = 0
    basePath = 'D:/dataset/BirdCLEF 2023/mk/part3/'

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver5.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver5.h5', 'a')

    Xdata, Ydata = audioFileLoad(basePath)
    X_train, X_test, y_train, y_test = dataToDivision(Xdata, Ydata)
    dataSet = [['trainSet', X_train, y_train], ['testSet', X_test, y_test]]

    for subset, X, Y in dataSet:
        # data_group = h5pyfile.create_group(subset)
        # dataX0 = data_group.create_dataset('X0', shape=(0, 256, 1024, 1), maxshape=(None, 256, 1024, 1), dtype=np.uint8)
        # labels = data_group.create_dataset( 'Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]), dtype=np.float32)

        for p, y in zip(X, Y):
            audio, sr = dataOpen(p)
            X0 = onPreprocessing(audio, sr=32000)

            for x in X0:
                # x = cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                x = np.expand_dims(x, axis=-1)

                # print(f'{x.shape}, {np.argmax(y)}_{num}.png')

                if subset == 'trainSet':
                    path_file = f'D:\\dataset\\BirdCLEF 2023\\mk\\img\\trainSet\\{np.argmax(y)}_{num}.png'
                else:
                    path_file = f'D:\\dataset\\BirdCLEF 2023\\mk\\img\\testSet\\{np.argmax(y)}_{num}.png'

                cv2.imwrite(path_file, x)
                num += 1

                # x = x.reshape((1,) + x.shape)
                # y = y.reshape((1,) + y.shape)
                #
                # current_data_size = dataX0.shape[0]
                # new_data_size = current_data_size + x.shape[0]
                # dataX0.resize(new_data_size, axis=0)
                # dataX0[current_data_size:new_data_size, :] = x
                #
                # current_data_size = labels.shape[0]
                # new_data_size = current_data_size + y.shape[0]
                # labels.resize(new_data_size, axis=0)
                # labels[current_data_size:new_data_size, :] = y
                #
                # print(f'{subset}] X: {dataX0.shape}, y: {labels.shape}')
