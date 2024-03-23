import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalMaxPooling2D
import math

df = pd.read_csv("training_solutions_rev1.csv")
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

print(df_train.describe())
print("Shape of training data-set : ", df_train.shape)
print("Shape of testing data-set : ", df_test.shape)

test_img_path = "images_test_rev1"
train_img_path = "images_training_rev1"


def plot_random_galaxy(path, sample=5):
    random_image = random.sample(os.listdir(path), sample)

    plt.figure(figsize=(16, 5))
    for i in range(sample):
        plt.subplot(1, sample, i + 1)
        img = tf.io.read_file(os.path.join(path, random_image[i]))
        img = tf.io.decode_image(img)
        plt.imshow(img)
        plt.title(f'Class: {random_image[i]}\nShape: {img.shape}')
        plt.axis(False)

# plot_random_galaxy(train_img_path)
# plt.show()

ORIG_SHAPE = (424, 424)
CROP_SIZE = (256, 256)
IMG_SHAPE = (64, 64)


def get_image(path, x1, y1, shape, crop_size):
    x = plt.imread(path)
    x = x[x1:x1 + crop_size[0], y1:y1 + crop_size[1]]
    x = resize(x, shape)
    x = x / 255.
    return x


def get_all_images(dataframe, shape=IMG_SHAPE, crop_size=CROP_SIZE):
    x1 = (ORIG_SHAPE[0] - CROP_SIZE[0]) // 2
    y1 = (ORIG_SHAPE[1] - CROP_SIZE[1]) // 2

    sel = dataframe.values
    ids = sel[:, 0].astype(int).astype(str)
    y_batch = sel[:, 1:]
    x_batch = []
    for i in tqdm(ids):
        x = get_image('/Users/lokranjan/PycharmProjects/galaxyclassifier/images_training_rev1/' + i + '.jpg', x1, y1,
                      shape=shape, crop_size=crop_size)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch, y_batch


X_train, y_train = get_all_images(df_train)
X_test, y_test = get_all_images(df_test)

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(512, (3, 3), input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(37))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy', root_mean_squared_error])

print(model.summary())

batch_size = 128
with strategy.scope():
    model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

