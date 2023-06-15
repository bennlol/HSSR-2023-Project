import tensorflow as tf
import numpy as np
from keras.utils import Sequence
import cv2, sys
import image_manipulation
import random
# from image_manipulation import *
class Siamese_data_gen(Sequence):
    def __init__(self, data_paths, class_data, batch_size, input_size, shuffle=True, data_augmentation = True):
        self.paths = data_paths 
        self.class_data = class_data
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.on_epoch_end()

    def __len__(self):
        if not self.data_augmentation: #If the data is being augmented double the amount of data else tell how many batches
            return int(np.ceil(len(self.paths)/1.0 / self.batch_size ))
        return int(np.ceil(len(self.paths)/1.0 / self.batch_size )) * 2

    def __getitem__(self, index):
        pairs = []
        labels = []
        data_augment = []
        for i in range(self.batch_size):
            idx1 = (index * self.batch_size + i) % (len(self.paths) *(self.data_augmentation+1))
            idx2 = (np.random.choice(len(self.paths)*(self.data_augmentation+1)))
            labels.append(int(self.class_data[idx1%len(self.paths)] == self.class_data[idx2%len(self.paths)]))
            pairs.append((self.paths[idx1%len(self.paths)], self.paths[idx2%len(self.paths)]))
            if self.data_augmentation:
                data_augment.append([idx1//len(self.paths), idx2//len(self.paths)])

        # Load images for pairs
        img1_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        img2_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        for i, pair in enumerate(pairs):
            img1 = tf.keras.preprocessing.image.load_img(pair[0], target_size=self.input_size, color_mode='grayscale')
            img2 = tf.keras.preprocessing.image.load_img(pair[1], target_size=self.input_size, color_mode='grayscale')
            img1 = cv2.resize(np.array(img1), self.input_size[0:2], interpolation = cv2.INTER_AREA)            
            img2 = cv2.resize(np.array(img2), self.input_size[0:2], interpolation = cv2.INTER_AREA)            
            if self.data_augmentation:
                img1 = self.apply_data_augmentation(img1) if data_augment[i][0] else img1
                img2 = self.apply_data_augmentation(img2) if data_augment[i][1] else img2
                
            img1_arr[i] = np.array(img1).reshape(self.input_size)
            img2_arr[i] = np.array(img2).reshape(self.input_size)
        return [img1_arr, img2_arr], np.array(labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.paths))
            np.random.shuffle(indices)
            self.paths = [self.paths[i] for i in indices]
            self.class_data = [self.class_data[i] for i in indices]
    
    def apply_data_augmentation(self, img):
        choose_augment = random.randint(0,2)
        if not choose_augment:
            return image_manipulation.randomRotation(img)
        elif choose_augment == 1:
            return image_manipulation.randomFlip(img)
        return image_manipulation.randomNoise(img)