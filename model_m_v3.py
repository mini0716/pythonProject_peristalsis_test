from imports_header import *


def residual_block(x, filters, kernel_size=3, strides=1):
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    if strides > 1:
        x = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)

    out = add([x, y])

    return Activation('relu')(out)


def resnet(input_iamge):
    x = Conv2D(64, (7, 7), strides=2, padding='same')(input_iamge)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        g = x = residual_block(x, filters, strides=2)

        for _ in range(1, 3):
            x = residual_block(x, filters)

        x = concatenate([x, g], axis=-1)

    x = GlobalAveragePooling2D()(x)

    return x


def basic_block(inputs, filters, spatial_stride, temporal_stride):
    x = Conv3D(filters, (1, 1, 1), strides=(1, spatial_stride, temporal_stride), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv3D(filters, (1, 1, 1), strides=(1, spatial_stride, temporal_stride), padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = ReLU()(x)
    return x


def slowfastnet(input_video):
    slow_path = Conv3D(64, (1, 7, 7), strides=(1, 2, 2), padding='same')(input_video)
    slow_path = BatchNormalization()(slow_path)
    slow_path = ReLU()(slow_path)
    slow_path = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(slow_path)
    slow_path = basic_block(slow_path, 64, spatial_stride=1, temporal_stride=1)

    fast_path = Conv3D(8, (5, 7, 7), strides=(1, 2, 2), padding='same')(input_video)
    fast_path = BatchNormalization()(fast_path)
    fast_path = ReLU()(fast_path)
    fast_path = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(fast_path)
    fast_path = basic_block(fast_path, 8, spatial_stride=1, temporal_stride=1)

    slow_path = basic_block(slow_path, 64, spatial_stride=1, temporal_stride=1)
    fast_path = basic_block(fast_path, 8, spatial_stride=1, temporal_stride=1)

    x = concatenate([slow_path, fast_path])
    x = GlobalAveragePooling3D()(x)

    return x


def onModel(input_shape1, num_classes):
    input_iamge_tensor, input_video_tensor = Input(shape=input_shape1), Input(shape=input_shape1)

    res_out, slow_out = resnet(input_iamge_tensor), slowfastnet(input_video_tensor)

    x = concatenate([res_out, slow_out])

    x = Dense(1024)(x)
    x = Dropout(0.25)(x)
    x = ReLU()(x)

    x = Dense(1024)(x)
    x = Dropout(0.25)(x)

    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_iamge_tensor, outputs=x)


def dataLoad(path=r'D:\dataset\BirdCLEF 2023\mk\NPY\BC_ver5.h5'):
    h5pyfile = h5py.File(path, 'r')
    subset = ['trainSet', 'testSet']

    trainSet, testSet = h5pyfile[subset[0]], h5pyfile[subset[1]]

    return trainSet['X0'], trainSet['Y'][:], testSet['X0'], testSet['Y'][:]


if __name__ == '__main__':
    batchS = 32
    trainX, trainY, testX, testY = dataLoad()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    # ReduceLROnPlateau 콜백 정의
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model = onModel(trainX.shape[1:], trainY.shape[1])
    y_train = np.argmax(trainY, axis=1)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[reduce_lr],
              epochs=200, validation_batch_size=batchS, batch_size=batchS, shuffle=False)
