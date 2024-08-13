from imports_header import *
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152


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
    x = UpSampling2D(size=(2, 2))(inputs)

    return x


def onBlock(I, nuit):
    x0 = Conv2D(nuit, (1, 1), activation='relu', padding='same')(I)
    x0 = Conv2D(nuit, (3, 3), activation='relu', padding='same')(x0)

    x1 = Conv2D(nuit, (1, 1), activation='relu', padding='same')(I)

    x = add([x0, x1])

    x = Conv2D(nuit, (3, 3), activation='relu', padding='same')(x)

    return x


def onModel(input_shape, num_classes):
    input_video_tensor = Input(shape=input_shape)
    x = tf.divide(input_video_tensor, 255)
    x = ReLU()(x)

    x = Conv2D(24, (7, 7), activation='relu', padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_video_tensor, outputs=x)


def dataLoad(path=r'D:/dataset/ISIC/NPY/ISIC_ver1.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X'], trainSet['Y'][:], testSet['X'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 200

    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    model = onModel(trainX.shape[1:], trainY.shape[1])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=batchS, shuffle=False)

    test_loss, test_acc = model.evaluate(testX, testY, batch_size=batchS)
    print(f'test_loss = {test_loss}, test_acc = {test_acc}')
