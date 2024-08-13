# 1, 2, 4, 8, 16, 61
from imports_header import *
from random import shuffle


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"파일 삭제 실패: {e}")


if __name__ == '__main__':
    PATHs = 'D:\\dataset\\BirdCLEF 2023\\mk\\img\\'
    subPaths = ['trainSet', 'testSet']

    delete_file(r'J:/BirdCLEF 2023/mk/NPY/BC_ver5.h5')
    h5pyfile = h5py.File('J:/BirdCLEF 2023/mk/NPY/BC_ver5.h5', 'a')
    classNumber = 247  # 총 클래스 수

    for subPath in subPaths:
        path = f'{PATHs}{subPath}'

        data_group = h5pyfile.create_group(subPath)
        dataX0 = data_group.create_dataset('X0', shape=(0, 640, 640, 3), maxshape=(None, 640, 640, 3), dtype=np.uint8)
        labels = data_group.create_dataset('Y', shape=(0, classNumber), maxshape=(None, classNumber), dtype=np.float32)

        fileNames = os.listdir(path)
        shuffle(fileNames)

        for fileName in fileNames:
            imgPath = f'{path}//{fileName}'

            img = cv2.imread(imgPath)
            numLabel = int(fileName.split('_')[0])

            X = img.reshape((1,) + img.shape)
            y = np.zeros((1, classNumber))
            y[:, numLabel] = 1

            current_data_size = dataX0.shape[0]
            new_data_size = current_data_size + X.shape[0]
            dataX0.resize(new_data_size, axis=0)
            dataX0[current_data_size:new_data_size, :] = X

            current_data_size = labels.shape[0]
            new_data_size = current_data_size + y.shape[0]
            labels.resize(new_data_size, axis=0)
            labels[current_data_size:new_data_size, :] = y

            print(f'{subPath}] X: {dataX0.shape}, y: {labels.shape}')
