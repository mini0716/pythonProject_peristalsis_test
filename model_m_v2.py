from imports_header import *


def conv3d_bn(x, filters, kernel_size, strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return MaxPooling2D((2, 2))(x)


def conv1d_bn(x, filters, kernel_size, strides=1, padding='same'):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def feature_extractor(input_shape, slowfast_split=3):
    input_tensor = Input(shape=input_shape)


    # Slow pathway
    slow_path = input_tensor[:, ::slowfast_split, :, :, :]  # Subsample frames
    for _ in range(5):
        slow_path = conv3d_bn(slow_path, 64, (1, 1, 1))
    slow_path = MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4))(slow_path)

    # Fast pathway
    fast_path = input_tensor
    for _ in range(5):
        fast_path = conv3d_bn(fast_path, 64, (1, 1, 1))
    fast_path = MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4))(fast_path)

    # Merge Slow and Fast pathways
    merged_path = concatenate([slow_path, fast_path], axis=1)

    # Common pathway
    for _ in range(4):
        merged_path = conv3d_bn(merged_path, 128, (1, 3, 3))
        merged_path = MaxPooling3D(pool_size=(1, 2, 2))(merged_path)

    merged_path = GlobalAveragePooling3D()(merged_path)

    # Fully connected layer
    output = Dense(247, activation='softmax')(merged_path)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output, name='SlowFast')
    return model


def feature_extractor_3D(X0, X1, X2, units=[64, 64, 64, 64, 64]):


    for _ in range(5):
        X0 = conv3d_bn(X0, units[_], (1, 1, 1), (1, 1, 1))
        X1 = conv3d_bn(X1, units[_], (1, 1, 1), (1, 1, 1))
        X2 = conv3d_bn(X2, units[_], (1, 1, 1), (1, 1, 1))

    merged_path = concatenate([MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 2, 2))(X0),
                               MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 2, 2))(X1),
                               MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 2, 2))(X2)], axis=1)

    for _ in range(4):
            merged_path = MaxPooling3D(pool_size=(1, 2, 2))(conv3d_bn(merged_path, 128, (1, 3, 3)))

    return GlobalAveragePooling3D()(merged_path)


def onModel(inputX0_shape, inputX1_shape, inputX2_shape, output_shape):
    inputX0, inputX1, inputX2 = Input(shape=inputX0_shape), Input(shape=inputX1_shape), Input(shape=inputX2_shape)

    x = feature_extractor_3D(inputX0, inputX1, inputX2)

    output = Dense(output_shape, activation='softmax')(x)

    return Model(inputs=(inputX0, inputX1, inputX2), outputs=output)


def dataLoad():
    h5pyfile = h5py.File(r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver3.h5', 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]
    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


if __name__ == '__main__':
# def myTraining():
    batchS = 5
    trainX0, trainY, testX0, testY = dataLoad()
    print(trainX0.shape, trainY.shape, testX0.shape, testY.shape)

    model = feature_extractor(trainX0.shape[1:], 247)

    y_train = np.argmax(trainY, axis=1)
    classWeights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX0, trainY, validation_data=(testX0, testY),
              epochs=200, validation_batch_size=batchS, batch_size=batchS, shuffle=False, class_weight=classWeights)

    #
    #