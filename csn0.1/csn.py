from keras.models import Sequential
import tensorflow as tf
from keras import Model
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from siamese_data_gen import Siamese_data_gen 
import numpy as np
import os

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
TRAINING_DIR = 'data\\train\\'
VALIDATION_DIR = "data\\validate\\"

epochs = 5
batch_size = 4
margin = 1

def load_data(dir):
    data_dir = os.path.join(DIR, dir)
    filelist = []
    classlist = []
    for root, dirpath, names in os.walk(data_dir):
        for name in names:
            soft_path = os.path.join(root, name)
            filelist.append(soft_path)
            classlist.append(root[-1])
    return filelist, classlist


def euclid_dist(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss

train_data_paths, train_class_data  = load_data(TRAINING_DIR)
val_data_paths, val_class_data = load_data(VALIDATION_DIR)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(800,800,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))cl
# model.add(Dropout(0.1))

# model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())
# model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))

input1=Input((800,800,1))
input2=Input((800,800,1))

tower1 = model(input1)
tower2 = model(input2)

merge_layer = Lambda(euclid_dist)([tower1, tower2])
normal_layer = BatchNormalization()(merge_layer)
output_layer = Dense(1, activation="sigmoid")(normal_layer)
siamese = Model(inputs=[input1, input2], outputs=output_layer)



siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])


traingen = Siamese_data_gen(train_data_paths, train_class_data, batch_size = batch_size, input_size=(800,800,1))
print(siamese.summary())
valgen = Siamese_data_gen(val_data_paths, val_class_data, batch_size = batch_size, input_size=(800,800,1))

#(path) (name, type)
history = siamese.fit(
    traingen,
    validation_data=valgen,
    batch_size=batch_size,
    epochs=epochs,
)