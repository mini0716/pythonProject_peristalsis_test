from imports_header import *


def basic_block(inputs, filters, spatial_stride=1, temporal_stride=1):
    x = Conv2D(filters, (1, 1), strides=(spatial_stride, temporal_stride), padding='same')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    x = concatenate([x, inputs], axis=-1)
    x = ReLU()(x)

    x = Conv2D(filters, (1, 1), strides=(spatial_stride, temporal_stride), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def encoder_block(inputs, unit):
    x = basic_block(inputs, unit)
    return x


def decoder_block(inputs):
    return UpSampling2D(size=(2, 2))(inputs)


def onModel(input_shape, num_classes):
    input_video_tensor = Input(shape=input_shape)
    x = tf.divide(input_video_tensor, 255)
    units = [18, 32, 64, 128]

    ### To ponder a problem. ###########################################################################################
    x = Conv2D(9, (3, 3), padding='same')(x)
    x = Conv2D(9, (3, 3), padding='same')(x)
    x = ReLU()(x)

    tPonder1 = encoder_block(x, units[0])
    samplin1 = MaxPooling2D((2, 2))(tPonder1)

    tPonder2 = encoder_block(samplin1, units[1])
    tPonder2 = encoder_block(tPonder2, units[1])
    samplin2 = MaxPooling2D((2, 2))(tPonder2)

    samplin2 = Conv2D(units[1], (3, 3), padding='same')(samplin2)
    samplin2 = Conv2D(units[1], (3, 3), padding='same')(samplin2)
    samplin2 = ReLU()(samplin2)

    tPonder3 = encoder_block(samplin2, units[2])
    tPonder3 = encoder_block(tPonder3, units[2])
    samplin3 = MaxPooling2D((2, 2))(tPonder3)

    samplin3 = Conv2D(units[2], (3, 3), padding='same')(samplin3)
    samplin3 = Conv2D(units[2], (3, 3), padding='same')(samplin3)
    samplin3 = ReLU()(samplin3)

    tPonder4 = encoder_block(samplin3, units[3])
    samplin4 = MaxPooling2D((2, 2))(tPonder4)

    tPonderFlatten = GlobalAveragePooling2D()(samplin4)

    ### To imagine a problem. ##########################################################################################
    tImagine1 = decoder_block(samplin1)
    tImagine1 = encoder_block(tImagine1, units[1])
    tImagineSampling1 = MaxPooling2D((2, 2))(tImagine1)

    tImagine2 = decoder_block(samplin2)
    tImagine2 = add([tImagine2, tImagineSampling1])
    tImagine2 = encoder_block(tImagine2, units[2])
    tImagineSampling2 = MaxPooling2D((2, 2))(tImagine2)

    tImagine3 = decoder_block(samplin3)
    tImagine3 = add([tImagine3, tImagineSampling2])
    tImagine3 = encoder_block(tImagine3, units[3])
    tImagineSampling3 = MaxPooling2D((2, 2))(tImagine3)

    tImagine4 = decoder_block(samplin4)
    tImagine4 = add([tImagine4, tImagineSampling3])

    tImagineFlatten = GlobalAveragePooling2D()(tImagine4)

    ### Merge data. ####################################################################################################
    x = concatenate([tPonderFlatten, tImagineFlatten])

    x = Dense(num_classes, activation='softmax')(x)  # 판단

    return Model(inputs=input_video_tensor, outputs=x)


def dataLoad(path=r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver5.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 1
    trainX, trainY = np.load(r'D:/dataset/ISIC/NPY/Xtrain.npy'), np.load(r'D:/dataset/ISIC/NPY/ytrain.npy')
    print(trainX.shape, trainY.shape)

    testX, testY = np.load(r'D:/dataset/ISIC/NPY/Xtest.npy'), np.load(r'D:/dataset/ISIC/NPY/ytest.npy')
    print(testX.shape, testY.shape)

    model = onModel(trainX.shape[1:], trainY.shape[1])
    y_train = np.argmax(trainY, axis=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[reduce_lr], epochs=200, validation_batch_size=batchS, batch_size=batchS, shuffle=False)
