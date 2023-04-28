from keras.models import Sequential
from keras import Model
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_1_DIR = 'data\\train1\\'
TRAINING_2_DIR = 'data\\train2\\'
VALIDATION_1_DIR = "data\\test1\\"
VALIDATION_2_DIR = "data\\test2\\"
epochs = 10
batch_size = 10
margin = 1

def load_data(dir):
    data_dir = os.path.join(DIR, dir)
    filelist = []
    for root, dirpath, names in os.walk(data_dir):
        for name in names:
            soft_path = os.path.join(root, name)
            filelist.append(soft_path)
    x = np.array([np.array(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in filelist])
    print(x.shape)
    return x


def euclid_dist(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss


x_train_1 = np.concatenate((load_data(TRAINING_1_DIR+'0\\'), load_data(TRAINING_1_DIR+'1\\')))
x_train_2 = np.concatenate((load_data(TRAINING_2_DIR+'0\\'), load_data(TRAINING_2_DIR+'1\\')))
labels_train = np.concatenate((np.zeros(28),np.ones(28)))

x_val_1 = np.concatenate((load_data(VALIDATION_1_DIR+'0\\'), load_data(VALIDATION_1_DIR+'1\\')))
x_val_2 = np.concatenate((load_data(VALIDATION_2_DIR+'0\\'), load_data(VALIDATION_2_DIR+'1\\')))
labels_val = np.concatenate((np.zeros(10),np.ones(10)))

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(800,800,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))


model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
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

history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)


# train_datagen = ImageDataGenerator(rescale=1.0/255.)
# train_generator = train_datagen.flow_from_directory([TRAINING_1_DIR,TRAINING_2_DIR],
#                             batch_size=batch_size,
#                             class_mode='binary',
#                             target_size=(200, 200),
#                             color_mode = 'grayscale'
# )

# validation_datagen = ImageDataGenerator(rescale=1.0/255.)
# validation_generator = validation_datagen.flow_from_directory([VALIDATION_1_DIR,VALIDATION_2_DIR],
#                             batch_size=batch_size,
#                             class_mode='binary',
#                             target_size=(200, 200),
#                             color_mode = 'grayscale')

# history = siamese.fit_generator(train_generator,
#                             epochs=epochs,
#                             verbose=1,
#                             validation_data=validation_generator)


################# ----------------------------------------------------------------------- ################
def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the contrastive loss
plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

# results = siamese.evaluate([x_test_1, x_test_2], labels_test)
# print("test loss, test acc:", results)
