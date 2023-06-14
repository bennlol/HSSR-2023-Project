import cv2, os, random, sys
import numpy as np
from keras.models import model_from_json
from keras import Model
from image_manipulation import randomRotation
import keras
from keras.layers import *
from keras.models import Sequential
from keras.regularizers import l2
import tensorflow as tf
margin = 1
input_size = (400,400,1)
inputShape = (400,400,1)

def load_pair(img1, img2, dir):
    img1 = cv2.imread(os.path.join(dir,img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(dir,img2), cv2.IMREAD_GRAYSCALE)
    img1 = np.expand_dims(img1, axis=-1)  # Add extra dimension
    img2 = np.expand_dims(img2, axis=-1)  # Add extra dimension
    img1 = cv2.resize(img1, (400, 400), interpolation = cv2.INTER_AREA)  # Resize image
    img2 = cv2.resize(img2, (400, 400), interpolation = cv2.INTER_AREA)  # Resize image
    return [img1, img2]

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
cluster_dirs = [os.path.join(os.path.join(DIR, "clusters"),str(dir)) for dir in range(0,6)] #use like cluster_dirs[0] for the 0 cluster


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


model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.4))

input1=Input(inputShape)
input2=Input(inputShape)

tower1 = model(input1)
tower2 = model(input2)

merge_layer = Lambda(euclid_dist)([tower1, tower2])
normal_layer = BatchNormalization()(merge_layer)
output_layer = Dense(1, activation="sigmoid")(normal_layer)
siamese = Model(inputs=[input1, input2], outputs=output_layer)

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])

siamese.load_weights("model.h5")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T8-31step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (same image), prediction: ${results}")

results = siamese.predict(cv2.imread(DIR+"V4-T8-31step10s.png", cv2.IMREAD_GRAYSCALE), randomRotation(cv2.imread(DIR+"V4-T8-31step10s.png", cv2.IMREAD_GRAYSCALE)), cluster_dirs[2])
print(f"(single void defect) : (same image but rotated), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T16-86step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (double void defect), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T7-79step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (grain defect), prediction: ${results}")
