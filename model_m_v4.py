from imports_header import *


def basic_block(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, inputs])
    x = ReLU()(x)

    return x


def encoder_block(inputs, unit):
    x = basic_block(inputs, unit)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return x


def decoder_block(inputs, unit):
    # x = UpSampling2D(size=(2, 2))(inputs)
    x = Conv2DTranspose(unit, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(inputs)
    # x = basic_block(x, unit)

    return x


def onModel(input_shape, num_classes):
    input_video_tensor = Input(shape=input_shape)
    x = tf.divide(input_video_tensor, 255)
    N, num = 4, 2

    units, botts = [num ** i for i in range(1, N + 1)], []
    print(units)

    for _ in range(N):
        if _ == 0:
            x = tf.divide(input_video_tensor, 255)

            x = Conv2D(units[_], (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        else:
            x = concatenate([x, lDecoderLayer, rEncoderLayer], axis=-1)
            x = Conv2D(units[_], (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        ### brain(1) ###################################################################################################
        lEncoderLayer = encoder_block(x, units[_])
        lDecoderLayer = ReLU()(decoder_block(lEncoderLayer, units[_]))

        botts.append(GlobalAveragePooling2D()(lDecoderLayer))
        ################################################################################################################

        ### brain(2) ###################################################################################################
        rDecoderLayer = ReLU()(decoder_block(x, units[_]))
        rEncoderLayer = encoder_block(rDecoderLayer, units[_])

        botts.append(GlobalAveragePooling2D()(rEncoderLayer))
        ################################################################################################################

    x = concatenate(botts)

    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_video_tensor, outputs=x)


def dataLoad(path=r'D:/dataset/ISIC/NPY/ISIC_ver1.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X'], trainSet['Y'][:], testSet['X'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 50

    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    model = onModel(trainX.shape[1:], trainY.shape[1])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=batchS, shuffle=False)

    test_loss, test_acc = model.evaluate(testX, testY, batch_size=batchS)
    print(f'test_loss = {test_loss}, test_acc = {test_acc}')
