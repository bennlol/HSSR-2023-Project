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
from matplotlib import pyplot as plt
margin = 1

input_size = (400,400,1)
inputShape = (400,400,1)

def load_pair(img1, dir1, img2 = None, dir2 = None, augment = None):
    if img2 and dir2:
        img1 = cv2.imread(os.path.join(dir1,img1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(dir2,img2), cv2.IMREAD_GRAYSCALE)
        img1 = np.expand_dims(img1, axis=-1)  
        img2 = np.expand_dims(img2, axis=-1)  
        img1 = cv2.resize(img1, (400,400), interpolation = cv2.INTER_AREA) 
        img2 = cv2.resize(img2, (400,400), interpolation = cv2.INTER_AREA) 
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return [img1, img2]
    else:
        img1 = cv2.imread(os.path.join(dir1,img1), cv2.IMREAD_GRAYSCALE)
        img1 = np.expand_dims(img1, axis=-1)  
        img1 = cv2.resize(img1, (400, 400), interpolation = cv2.INTER_AREA)
        if augment:
            img1 = augment(img1)
        img1 = np.expand_dims(img1, axis=0)
        return img1

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

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0015)))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0015)))
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


siamese.load_weights(os.path.join("csn0.2","train on all data without shuffle before validation split","model.h5"))

results = siamese.predict(load_pair("V4-T8-31step10s.png", cluster_dirs[2], "V4-T8-31step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (same image), prediction: ${results}")


# results = siamese.predict([load_pair("V4-T8-31step10s.png", cluster_dirs[2]), load_pair("V4-T8-31step10s.png",cluster_dirs[2], augment=randomRotation)])
# print(f"(single void defect) : (random rotation), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png", cluster_dirs[2], "V4-T8-47step10s.png", cluster_dirs[2]))
print(f"(single void defect) : (different single void defect), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png", cluster_dirs[2], "V4-T16-86step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (double void defect), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png", cluster_dirs[2], "V4-T4-6step10s.png",cluster_dirs[2]))
print(f"(single void defect) : (grain defect), prediction: ${results}")

results = siamese.predict(load_pair("V1-T1-16step5s.png", cluster_dirs[5], "V1-T4-53step10s.png",cluster_dirs[5]))
print(f"(amorphous class 5) : (amorphous class 5), prediction: ${results}")

results = siamese.predict(load_pair("V3-T19-9step10s.png", cluster_dirs[4], "V1-T4-53step10s.png",cluster_dirs[5]))
print(f"(amorphous class 4) : (amorphous class 5), prediction: ${results}")

results = siamese.predict(load_pair("V1-T25-0step10s.png", cluster_dirs[0], "V4-T4-6step10s.png",cluster_dirs[2]))
print(f"(crystaline) : (grain defect), prediction: ${results}")