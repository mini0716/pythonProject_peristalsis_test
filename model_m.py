from imports_header import *


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv1d_bn(x, filters, kernel_size, strides=1, padding='same'):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def feature_extractor(input_shape1, input_shape2, slowfast_split=5):
    input_tensor1 = Input(shape=input_shape1)
    input_tensor2 = Input(shape=input_shape2)

    # Slow pathway
    slow_path = input_tensor1[:, ::slowfast_split, :, :, :]  # Subsample frames
    for _ in range(5):
        slow_path = conv3d_bn(slow_path, 64, (1, 1, 1))
    slow_path = MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4))(slow_path)

    # Fast pathway
    fast_path = input_tensor1
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

    model = tf.keras.models.Model(inputs=(input_tensor1, input_tensor2), outputs=output, name='SlowFast')
    return model


def feature_extractor_2D(x, slowfast_split=5, units=[64, 64, 64, 64, 64]):
    slow_path, fast_path = x[:, ::slowfast_split, :], x
    slow_path0, slow_path1 = x[:, ::slowfast_split-1, :], x[:, ::slowfast_split-2, :]

    for _ in range(5):
        fast_path, slow_path = conv2d_bn(fast_path, units[_], (1, 1)), conv2d_bn(slow_path, units[_], (1, 1))
        slow_path0, slow_path1 = conv2d_bn(slow_path0, units[_], (1, 1, 1)), conv2d_bn(slow_path1, units[_], (1, 1))

    merged_path = concatenate([MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(slow_path),
                               MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(fast_path),
                               MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(slow_path0),
                               MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(slow_path1)], axis=1)

    for _ in range(4):
        merged_path = MaxPooling2D(pool_size=(2, 2))(conv2d_bn(merged_path, 128, (3, 3)))

    return GlobalAveragePooling3D()(merged_path)



def feature_extractor_1D(x, slowfast_split=5, units=[64, 64, 64, 64, 64]):
    # x = tf.expand_dims(x, axis=-1)
    slow_path, fast_path = x[:, ::slowfast_split, :], x

    for _ in range(5):
        fast_path, slow_path = conv1d_bn(fast_path, units[_], 1, 4), conv1d_bn(slow_path, units[_], 1, 4)

    merged_path = concatenate([MaxPooling1D(pool_size=4, strides=4)(slow_path),
                               MaxPooling1D(pool_size=4, strides=4)(fast_path)], axis=1)

    for _ in range(4):
        merged_path = MaxPooling1D(pool_size=2)(conv1d_bn(merged_path, 128, 3))

    return GlobalAveragePooling1D()(merged_path)


def feature_extractor_3D(x, slowfast_split=5, units=[64, 64, 64, 64, 64]):
    slow_path, fast_path = x[:, ::slowfast_split, :], x

    for _ in range(5):
        fast_path, slow_path = conv2d_bn(fast_path, units[_], (1, 1)), conv2d_bn(slow_path, units[_], (1, 1))

    merged_path = concatenate([MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4))(slow_path),
                               MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4))(fast_path)], axis=1)

    for _ in range(4):
        merged_path = MaxPooling3D(pool_size=(1, 2, 2))(conv3d_bn(merged_path, 128, (1, 3, 3)))

    return GlobalAveragePooling3D()(merged_path)


def onModel(inputSpectrogram_shape, output_shape):
    inputSpectrogram = Input(shape=inputSpectrogram_shape)

    x = feature_extractor_3D(inputSpectrogram)
    output = Dense(output_shape, activation='softmax')(x)

    return Model(inputs=inputSpectrogram, outputs=output)


def dataLoad():
    h5pyfile = h5py.File(r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver1.h5', 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]
    return trainSet['X1'], trainSet['Y'][:], testSet['X1'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 3
    trainSpectrogram, trainY, testSpectrogram, testY = dataLoad()
    print(trainSpectrogram.shape, trainY.shape, testSpectrogram.shape, testY.shape)

    model = onModel((25, 224, 224, 1), 247)

    y_train = np.argmax(trainY, axis=1)
    classWeights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainSpectrogram, trainY, validation_data=(testSpectrogram, testY), epochs=200, batch_size=batchS,
              shuffle=False, validation_batch_size=batchS, class_weight=classWeights)

