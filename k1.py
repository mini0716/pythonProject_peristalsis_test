import os
import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


def createPaths(basic_path):
    fileNames = [name for name in os.listdir(basic_path)]

    PATHS = [basic_path + name + '//' for name in fileNames]

    for PATH in PATHS:
        audioNames = os.listdir(PATH)
        paths = [PATH + audioName for audioName in audioNames if 'ogg' in audioName]

    return paths


def audioFileLoad(path):
    from tensorflow.keras.utils import to_categorical
    files, labels = [], []

    for label, subNames in enumerate(os.listdir(path)):
        for file in os.listdir(f'{path}/{subNames}'):
            if file.endswith('.ogg'):
                files.append(os.path.join(path, subNames, file))
                labels.append(label)

    return np.array(files), to_categorical(labels)


def signalToMelspectrogram(y):
    # Mel 스펙트로그램 계산
    S = librosa.feature.melspectrogram(y=y, sr=32000, n_fft=2048, hop_length=512, n_mels=128,
                                       fmin=40, power=2.0, fmax=15000)

    # 로그 스케일 변환 (dB 스케일)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=100)

    return S_db


def mKmean(X, n_clusters=2):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)

    return kmeans.labels_, n_clusters


def group_frequencies_into_bands(frequencies, avg_magnitude):
    # 주파수 대역 정의 (예: 0-500Hz, 500-1000Hz, ...)
    bands = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    band_magnitudes = np.zeros(len(bands) - 1)
    for i in range(len(bands) - 1):
        band_mask = (frequencies >= bands[i]) & (frequencies < bands[i + 1])
        if np.any(band_mask):
            band_magnitudes[i] = np.mean(avg_magnitude[band_mask])

    average_frequency_band = bands[np.argmax(band_magnitudes)] if bands[np.argmax(band_magnitudes)] != 0 else bands[np.argmax(band_magnitudes)+1]

    return average_frequency_band


def sizeMatching(X, __MAX_SHAPE_SIZE__):
    while (X.shape[1] < __MAX_SHAPE_SIZE__):
        X = np.hstack([X, X])
    return X[:, :__MAX_SHAPE_SIZE__]


def dataToDivision(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


def onPreprocessing(audio_signal, sr):
    # STFT 계산
    n_fft, hop_length = 2048, 512
    stft = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)

    # 절대값과 위상 구하기
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # 고주파 대역 추출 (평균 주파수 대역폭을 계산하여 적용)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 평균적으로 사용되는 주파수 대역 찾기
    # average_frequency_band = group_frequencies_into_bands(freq_bins, np.mean(magnitude, axis=1))

    # bin_2k = np.argmin(np.abs(freq_bins - average_frequency_band))

    K = 3
    freq_bins_k, N = mKmean(freq_bins, K)

    # freq_bins_0, freq_bins_1, freq_bins_2 = freq_bins.copy(), freq_bins.copy(), freq_bins.copy()
    # freq_bins_0, freq_bins_1, freq_bins_2 = freq_bins_0[freq_bins_k == 0], freq_bins_1[freq_bins_k == 1], freq_bins_2[
    #     freq_bins_k == 1]

    stft_low, stft_mid, stft_high = magnitude[freq_bins_k == 0] * np.exp(1j * phase[freq_bins_k == 0]), magnitude[
        freq_bins_k == 1] * np.exp(1j * phase[freq_bins_k == 1]), magnitude[freq_bins_k == 2] * np.exp(
        1j * phase[freq_bins_k == 2])

    y_low, y_mid, y_high = librosa.istft(stft_low, hop_length=hop_length), librosa.istft(stft_mid,
                                                                                         hop_length=hop_length), librosa.istft(
        stft_high, hop_length=hop_length)

    mel_low, mel_mid, mel_high = signalToMelspectrogram(y_low), signalToMelspectrogram(y_mid), signalToMelspectrogram(
        y_high)
    mel_low, mel_mid, mel_high = sizeMatching(mel_low, __MAX_SHAPE_SIZE__), sizeMatching(mel_mid,
                                                                                         __MAX_SHAPE_SIZE__), sizeMatching(
        mel_high, __MAX_SHAPE_SIZE__)
    mel_low, mel_mid, mel_high = mel_low.reshape(mel_low.shape + (1,)), mel_mid.reshape(
        mel_mid.shape + (1,)), mel_high.reshape(mel_high.shape + (1,))

    # print(mel_low.shape, mel_mid.shape, mel_high.shape)

    return mel_low, mel_mid, mel_high




if __name__ == '__main__':
    # audioPaths = createPaths(r'D://dataset//BirdCLEF 2023//mk//part3//')
    # __MAX_SHAPE_SIZE__ = 5573904
    __MAX_SHAPE_SIZE__ = 7500

    delete_file(r'D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver4.h5')
    h5pyfile = h5py.File('D:/dataset/BirdCLEF 2023/mk/NPY/BC_ver4.h5', 'a')

    Xdata, Ydata = audioFileLoad(r'D://dataset//BirdCLEF 2023//mk//part3//')
    audioPaths_X_train, audioPaths_X_test, audioPaths_y_train, audioPaths_y_test = dataToDivision(Xdata, Ydata)
    dataSet = [['trainSet', audioPaths_X_train, audioPaths_y_train], ['testSet', audioPaths_X_test, audioPaths_y_test]]

    for subset, X, Y in dataSet:
        data_group = h5pyfile.create_group(subset)
        dataX0 = data_group.create_dataset('X0', shape=(0, 128, 7500, 1), maxshape=(None, 128, 7500, 1), dtype=np.uint8)
        dataX1 = data_group.create_dataset('X1', shape=(0, 128, 7500, 1), maxshape=(None, 128, 7500, 1), dtype=np.uint8)
        dataX2 = data_group.create_dataset('X2', shape=(0, 128, 7500, 1), maxshape=(None, 128, 7500, 1), dtype=np.uint8)
        labels = data_group.create_dataset('Y', shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]), dtype=np.float32)

        for p, y in zip(X, Y):
            # audio, sr = dataOpen(p)
            audio_signal, sr = librosa.load(p)

            X0, X1, X2 = onPreprocessing(audio_signal, sr)

            X0, X1, X2 = X0.reshape((1,) + X0.shape), X1.reshape((1,) + X1.shape), X2.reshape((1,) + X2.shape)
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

            print(f'{subset}] X0: {dataX0.shape}, X1: {dataX1.shape}, X2: {dataX2.shape}, y: {labels.shape}')
