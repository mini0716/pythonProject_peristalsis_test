from imports_header import *


def basic_block(inputs, filters):
    x0 = Conv2D(filters, (1, 1), activation='relu', padding='same')(inputs)
    x1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)

    # x = concatenate([x0, x1])
    x = add([x0, x1])
    x = Conv2D(filters, (5, 5), activation='relu', padding='same')(x)

    return x


def encoder_block(inputs, unit):
    k = unit // 5
    x = basic_block(inputs, unit)
    x = basic_block(x, unit+k*1)
    x = Conv2D(unit+k*2, (3, 3), activation='relu', padding='same')(x)

    x = basic_block(x, unit+k*3)
    x = Conv2D(unit+k*4, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    return x


def decoder_block(inputs, unit):
    k = unit // 5

    x = UpSampling2D((2, 2))(inputs)

    x = Conv2D(unit+k, (3, 3), activation='relu', padding='same')(x)
    x = basic_block(x, unit+k*2)

    x = Conv2D(unit+k*3, (3, 3), activation='relu', padding='same')(x)
    x = basic_block(x, unit+k*4)
    x = basic_block(x, unit+k*5)

    return x


def divideLayer(inputs, w, h):
    x = inputs

    v0 = x[:, 0:w//2, 0:h//2, :]
    v1 = x[:, 0:w//2, h//2:h, :]
    v2 = x[:, w//2:w, 0:h//2, :]
    v3 = x[:, w//2:w, h//2:h, :]

    return v0, v1, v2, v3


def dataLoad(path=r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver5.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


def brainStrecture(I, units, t=True):
    arrEncode = []
    arrDecode = []

    for unit in units:
        if t:
            encode = decode = I
            decode = MaxPooling2D((2, 2))(decode)
            t = False
        else:
            encode, decode = decode, encode
            encode = MaxPooling2D((2, 2))(encode)
            decode = MaxPooling2D((2, 2))(decode)

        encode = encoder_block(encode, unit)
        decode = decoder_block(decode, unit)
        encode = Conv2D(unit, (3, 3), activation='relu', padding='same')(encode)
        decode = Conv2D(unit, (3, 3), activation='relu', padding='same')(decode)

        arrEncode.append(GlobalAveragePooling2D()(encode))
        arrDecode.append(GlobalAveragePooling2D()(decode))

    return concatenate(arrEncode), concatenate(arrDecode)


def onModel(input_shape, num_classes):
    x = input_tensor = Input(shape=input_shape)
    x = tf.divide(x, 255)
    units = [8, 16, 32, 64, 128]

    v0, v1, v2, v3 = divideLayer(x, 224, 224)
    x = brainStrecture(x, units)
    v0 = brainStrecture(v0, units)
    v1 = brainStrecture(v1, units)
    v2 = brainStrecture(v2, units)
    v3 = brainStrecture(v3, units)

    x = concatenate([x[0], x[1],
                     v0[0], v0[1],
                     v1[0], v1[1],
                     v2[0], v2[1],
                     v3[0], v3[1]
                     ])

    x = Flatten()(x)

    x = Dense(num_classes, activation='softmax')(x)  # 판단

    return Model(inputs=input_tensor, outputs=x)


if __name__ == '__main__':
    batchS = 16
    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    model = onModel(trainX.shape[1:], trainY.shape[1])
    y_train = np.argmax(trainY, axis=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[reduce_lr], epochs=200, validation_batch_size=batchS, batch_size=batchS, shuffle=False)


