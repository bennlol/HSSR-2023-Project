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
        triplets = []
        data_augment = []
        for i in range(self.batch_size):
            neg_index = None #neg anch and pos are index for the anchour positivve and negative
            anch_index = (index * self.batch_size + i) % (len(self.paths) *(self.data_augmentation+1))
            pos_index = (np.random.choice(len(self.paths)*(self.data_augmentation+1)))
            while self.class_data[anch_index%len(self.paths)] != self.class_data[pos_index%len(self.paths)]:
                if neg_index == None:
                    neg_index = pos_index
                pos_index = (np.random.choice(len(self.paths)*(self.data_augmentation+1)))
            while neg_index == None or self.class_data[anch_index%len(self.paths)] == self.class_data[neg_index%len(self.paths)]:
                neg_index =  (np.random.choice(len(self.paths)*(self.data_augmentation+1)))
                
            triplets.append((self.paths[anch_index%len(self.paths)], self.paths[pos_index%len(self.paths)], self.paths[neg_index%len(self.paths)]))
            if self.data_augmentation:
                data_augment.append([anch_index//len(self.paths), pos_index//len(self.paths), neg_index//len(self.paths)])

        # Load images for pairs
        anchour_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        positive_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        negative_arr = np.zeros((self.batch_size,) + self.input_size, dtype="float32")
        for i, triplet in enumerate(triplets):
            anch = tf.keras.preprocessing.image.load_img(triplet[0], target_size=self.input_size, color_mode='grayscale')
            pos = tf.keras.preprocessing.image.load_img(triplet[1], target_size=self.input_size, color_mode='grayscale')
            neg = tf.keras.preprocessing.image.load_img(triplet[1], target_size=self.input_size, color_mode='grayscale')            
            anch = cv2.resize(np.array(anch), self.input_size[0:2], interpolation = cv2.INTER_AREA)            
            pos = cv2.resize(np.array(pos), self.input_size[0:2], interpolation = cv2.INTER_AREA)            
            neg = cv2.resize(np.array(neg), self.input_size[0:2], interpolation = cv2.INTER_AREA)            
            if self.data_augmentation:
                anch = self.apply_data_augmentation(anch) if data_augment[i][0] else anch
                pos = self.apply_data_augmentation(pos) if data_augment[i][1] else pos
                neg = self.apply_data_augmentation(neg) if data_augment[i][1] else neg
                
            anchour_arr[i] = np.array(anch).reshape(self.input_size)
            positive_arr[i] = np.array(pos).reshape(self.input_size)
            negative_arr[i] = np.array(neg).reshape(self.input_size)
        return [anchour_arr, positive_arr, negative_arr]
    
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