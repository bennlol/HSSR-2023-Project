import tensorflow as tf
import numpy as np
from keras.utils import Sequence

class Siamese_data_gen(Sequence):
    def __init__(self, data_paths, class_data, batch_size, input_size, shuffle=True):
        self.paths = data_paths
        self.class_data = class_data
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths)/1.0 / self.batch_size ))

    def __getitem__(self, index):
        pairs = []
        labels = []
        for i in range(self.batch_size):
            idx1 = (index * self.batch_size + i) % len(self.paths)
            idx2 = (np.random.choice(len(self.class_data))) % len(self.paths)
            pairs.append((self.paths[idx1], self.paths[idx2]))
            labels.append(int(self.class_data[idx1] == self.class_data[idx2]))

        # Load images for pairs
        img1_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        img2_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        for i, pair in enumerate(pairs):
            img1 = tf.keras.preprocessing.image.load_img(pair[0], target_size=self.input_size, color_mode='grayscale')
            img2 = tf.keras.preprocessing.image.load_img(pair[1], target_size=self.input_size, color_mode='grayscale')
            img1_arr[i] = np.array(img1).reshape(self.input_size)
            img2_arr[i] = np.array(img2).reshape(self.input_size)
        return [img1_arr, img2_arr], np.array(labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.paths))
            np.random.shuffle(indices)
            self.paths = [self.paths[i] for i in indices]
            self.class_data = [self.class_data[i] for i in indices]