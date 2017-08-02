import random
import threading
from os import listdir

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from scipy import misc
import cv2
import matplotlib.pyplot as plt


def build_alexnet():

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

    # AlexNet with batch normalization in Keras

    model = Sequential()
    model.add(Convolution2D(64, (11, 11),
                            padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, (7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1000, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, kernel_initializer='normal', activation='tanh'))

    return model


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
# write the definition of your data generator
def myGenerator(img_dir, ant_path, batch_size):

    # List all files in directory
    filenames = [f for f in listdir(img_dir)]
    filenames.sort()
    # Read annotation into a list
    with open(ant_path) as ant:
        annotation = ant.readlines()
        annotation = [x.strip() for x in annotation]

    # Shuffle image filenames
    random.seed(1234)
    random.shuffle(filenames)
    random.seed(1234)
    random.shuffle(annotation)

    # Load mean image for mean subtraction
    mean = cv2.imread(MEAN_IMAGE)
    mean = cv2.cvtColor(mean, cv2.COLOR_BGR2RGB)

    while 1:
        for batch in range(len(filenames) / batch_size):
            x_batch = np.zeros((batch_size, 210, 280, 3), dtype=np.float32)
            y_batch = np.zeros((batch_size, output_dim), dtype=np.float32)
            j = batch * batch_size

            for i in range(batch_size):
                im = cv2.imread(img_dir + filenames[j + i])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                x_batch[i] = im - mean
                y_batch[i] = annotation[j + i].split(',')[2:]

            x_batch /= 255.
            yield x_batch, y_batch


class SaveModel(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
	return

    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
	self.model.save_weights('./epoch' + str(epoch) + '.hdf5')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


TRAIN_IMAGES = '/D/GTA_data/train/data/'
TRAIN_ANNOT = '/D/GTA_data/train/annotation_scale.txt'

TEST_IMAGES = '/D/GTA_data/test/data/'
TEST_ANNOT = '/D/GTA_data/test/annotation_scale.txt'

VALID_IMAGES = '/D/GTA_data/valid/data/'
VALID_ANNOT = '/D/GTA_data/valid/annotation_scale.txt'

MEAN_IMAGE = '/D/GTA_data/train/mean.bmp'
NUM_LABELS = 8

batch_size = 32
input_shape = (210, 280, 3)
output_dim = 8
num_epoch = 100

train_files = [f for f in listdir(TRAIN_IMAGES)]
num_steps = len(train_files) / batch_size

#test_files = [f for f in listdir(TEST_IMAGES)]
#test_num_steps = len(test_files) / batch_size

valid_files = [f for f in listdir(VALID_IMAGES)]
val_num_steps = len(valid_files) / batch_size

#best_weights_filepath = './best_weights.hdf5'
#earlyStopping = keras.callbacks.EarlyStopping(
#    monitor='val_loss', patience=5, verbose=1, mode='auto')
#saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
#                                                verbose=1, save_best_only=True, mode='auto')
saveModel = SaveModel()

# Build model
model = build_alexnet()
# Choose optimizer
adam = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')
# Load weights
# model.load_weights('./best_weights.hdf5')
model.load_weights('./temp.hdf5')

model.fit_generator(myGenerator(TRAIN_IMAGES, TRAIN_ANNOT, batch_size),
                    num_steps, epochs=num_epoch, verbose=1,
                    validation_data=None,
                    validation_steps=None, max_queue_size=100,
                    workers=4, use_multiprocessing=False, initial_epoch=1,
                    callbacks=[saveModel])

