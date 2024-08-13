import numpy as np
from imports_header import *


def onModel(input_shape, num_classes):

    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import ResNet50, ResNet101, EfficientNetB0
    from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

    # ResNet 모델을 만들기
    input_tensor = Input(shape=input_shape)
    resnet_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet')

    x = resnet_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(num_classes, activation='softmax')(x)  # 출력 계층

    return Model(inputs=input_tensor, outputs=x)


def dataLoad(path=r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver5.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 12
    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    model = onModel(trainX.shape[1:], trainY.shape[1])
    y_train = np.argmax(trainY, axis=1)

    # ReduceLROnPlateau 콜백 정의
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[reduce_lr],
              epochs=200, validation_batch_size=batchS, batch_size=batchS, shuffle=False)













