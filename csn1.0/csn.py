from keras.models import Sequential
import tensorflow as tf
from keras import Model
from keras.layers import *
from keras.regularizers import l2
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from siamese_data_gen import Siamese_data_gen
import numpy as np
import os
import sys

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = 'clusters'

epochs = 4
batch_size = 16
margin = 1
inputShape = (400,400,1)

def load_data(dir, split = (85,15), data_percent_used = 100):
    data_dir = os.path.join(DIR, dir)
    filelist = []
    classlist = []
    for root, dirpath, names in os.walk(data_dir):
        for name in names:
            if (root[-1] == "1" or root[-1] == "0") and name[-3:]=='png':
                soft_path = os.path.join(root, name)
                filelist.append(soft_path)
                classlist.append(root[-1])
    indices = np.arange(len(filelist))
    np.random.shuffle(indices)
    filelist = [filelist[i] for i in indices]
    classlist = [classlist[i] for i in indices]
    
    filelist = filelist[:int(len(filelist) * data_percent_used/100)]
    classlist = classlist[:int(len(classlist) * data_percent_used/100)]
    
    split_index = int(split[0]/100 * len(filelist)) 
    return filelist[:split_index], classlist[:split_index], filelist[split_index:], classlist[split_index:]


def euclid_dist(vects):
    anchor, positive, negative = vects
    positive_distance = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)
    negative_distance = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)
    return positive_distance, negative_distance


def loss(margin=1):
    def triplet_loss(y_pred, margin=1):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        distance_positive = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)
        distance_negative = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)
        loss = tf.math.maximum(distance_positive - distance_negative + margin, 0)
        return tf.math.reduce_mean(loss)

    return triplet_loss


train_data_paths, train_class_data, val_data_paths, val_class_data = load_data(DATA_DIR, split = (85,15), data_percent_used = 100)

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
input3=Input(inputShape)

tower1 = model(input1)
tower2 = model(input2)
tower3 = model(input3)

merge_layer = Lambda(euclid_dist)([tower1, tower2, tower3])
output_layer = merge_layer
siamese = Model(inputs=[input1, input2, input3], outputs=output_layer)

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
print(siamese.summary())

earlyStopping = tf.keras.callbacks.EarlyStopping(patience = 1)

traingen = Siamese_data_gen(train_data_paths, train_class_data, batch_size = batch_size, input_size=inputShape, shuffle = True)
valgen = Siamese_data_gen(val_data_paths, val_class_data, batch_size = batch_size, input_size=inputShape, shuffle=True, data_augmentation = False)

#(path) (name, type)
history = siamese.fit(
    traingen,
    validation_data=valgen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks = [earlyStopping]
)

siamese_json = siamese.to_json()
with open("model.json", "w") as json:
    json.write(siamese_json)
siamese.save_weights("model.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()