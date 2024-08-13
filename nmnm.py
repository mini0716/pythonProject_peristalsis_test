from imports_header import *

def dataLoad(path=r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver4.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


if __name__ == '__main__':
    # 데이터 증식을 위한 설정
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


    orgPath = f'D://dataset//BirdCLEF 2023//mk//img//trainSet//ORIGINAL//'
    augPath = f'D://dataset//BirdCLEF 2023//mk//img//trainSet//AUG//'

    batchS = 64
    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    trainY = np.argmax(trainY, axis=-1)
    testY = np.argmax(testY, axis=-1)

    cnt = 0
    for x, y in zip(trainX, trainY):
        cv2.imwrite(f'{orgPath}{cnt}_{y}.jpg', x)
        cnt += 1

        x = x.reshape((1,) + x.shape)

        # 이미지를 50개로 증식하고 저장
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augPath, save_prefix=f'{cnt}_{y}_aug', save_format='jpg'):
            i += 1
            if i >= 50:
                break  # 무한 루프 방지


    cnt = 0
    for x, y in zip(testX, testY):
        cv2.imwrite(f'D://dataset//BirdCLEF 2023//mk//img//testSet//{cnt}_{y}.jpg', x)
        cnt += 1

        # x = x.reshape((1,) + x.shape)
        #
        # # 이미지를 50개로 증식하고 저장
        # i = 0
        # for batch in datagen.flow(x, batch_size=1, save_to_dir=augPath, save_prefix=f'{cnt}_{y}_aug', save_format='jpeg'):
        #     i += 1
        #     if i >= 50:
        #         break  # 무한 루프 방지