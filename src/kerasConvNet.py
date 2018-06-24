import json
import os

import keras
import numpy as np
from PIL import Image
from keras import backend
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from definitions import RESOURCES_DIR
from image.mnistFormat import convertImageToMnistStandart


def test(model, testImageFolderPath, groundTruthLabels):
    onlyFileNames = [f for f in os.listdir(testImageFolderPath) if os.path.isfile(os.path.join(testImageFolderPath, f))]
    print(onlyFileNames)

    onlyfiles = [os.path.join(testImageFolderPath, f) for f in onlyFileNames]

    images = []
    for file in onlyfiles:
        gray = convertImageToMnistStandart(file)
        testImageArray = np.expand_dims(gray, axis=3)
        images.append(testImageArray)

    imagesStack = np.vstack([images])

    imagesStack = normalize_data(imagesStack)

    predictedLabels = model.predict(imagesStack, verbose=0).argmax(axis=-1)
    print("Found label: %s" % predictedLabels)

    groundTruthLabelsList = [groundTruthLabels.get(f) for f in onlyFileNames]
    print(groundTruthLabelsList)

    labels = keras.utils.to_categorical(groundTruthLabelsList, 10)
    score = model.evaluate(imagesStack, labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def normalize_data(images):
    images = images.astype('float32')
    images /= 255
    return images


def reshapeImage(x_train, imageShape):
    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], imageShape[0], imageShape[1], imageShape[2])
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    return x_train


def testOnMnistData(model, num_classes=10):
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = normalize_data(x_test)
    x_test = reshapeImage(x_test, imageShape)

    im = Image.fromarray(x_test)
    im.show()

    y_test = keras.utils.to_categorical(y_test, num_classes)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def train(model, batch_size=128, num_classes=10, epochs=12):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = reshapeImage(x_train, imageShape)
    x_test = reshapeImage(x_test, imageShape)

    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    weigthsPath = os.path.join(RESOURCES_DIR, "weights_epoch-" + str(epochs) + ".h5")
    model.save_weights(weigthsPath)
    print("Saved weights to resources folder")


def createModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def createImageShape(img_rows, img_cols):
    if backend.image_data_format() == 'channels_first':
        # print("channels_first")
        input_shape = (1, img_rows, img_cols)
    else:
        # print("channels_last")
        input_shape = (img_rows, img_cols, 1)
    return input_shape


def loadDictionaryFromJsonFile(groundTruthLabelsPath):
    with open(groundTruthLabelsPath, "r") as f:
        json_data = f.read()
        return json.loads(json_data)


if __name__ == '__main__':
    # input image dimensions
    img_rows, img_cols = 28, 28
    imageShape = createImageShape(img_rows, img_cols)

    model = createModel(imageShape, num_classes=10)
    # train(model, epochs=12)

    weigthsPath = os.path.join(RESOURCES_DIR, "weights_epoch-12.h5")
    model.load_weights(weigthsPath)

    testImageFolderPath = os.path.join(RESOURCES_DIR, "separate_numbers")
    groundTruthLabelsPath = os.path.join(RESOURCES_DIR, "separate_number_labels.json")

    groundTruthLabels = loadDictionaryFromJsonFile(groundTruthLabelsPath)

    test(model, testImageFolderPath, groundTruthLabels)

    # testOnMnistData(model)
