import tensorflow as tf
import cv2, numpy as np, random

class Siamese_data_gen(tf.keras.utils.Sequence):
    def __init__(self, dfx, dfy, batch_size, input_size = (800,800,1), shuffle = True):
        self.df = dfx.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.dfy = dfy.copy() 
           
    def __len__(self):
        return self.n
    
    def __get_input(self, path):
        image = tf.keras.utils.load_img(path, color_mode='grayscale')
        image = tf.keras.utils.img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        # image = np.expand_dims(image, axis=-1)
        # print(image.shape)
        return image
    
    def __get_output(self, label, num_classes):
        
        return tf.keras.utils.to_categorical(int(label), num_classes = len(num_classes))
    
    def __get_data(self, path_batch, types_batch):
        x_batch = np.asarray([self.__get_input(path) for path in path_batch])
        indexes = [random.randint(0,len(self.df)-1) for _ in range(len(path_batch))]
        path_batch_pair = [self.df[index] for index in indexes]
        types_batch_pair = [self.dfy[index] for index in indexes]
        x_batch_pair = np.asarray([self.__get_input(path) for path in path_batch_pair])
        
        y0_batch = np.asarray([self.__get_output(y,self.dfy) for y in types_batch])
        y0_batch_pair = np.asarray([self.__get_output(y,self.dfy) for y in types_batch_pair])
        y_batch = tuple([self.__get_output(int(types_batch[i]==types_batch_pair[i]), self.dfy ) for i in range(len(types_batch))])
        return [tuple([x_batch[i], x_batch_pair[i]]) for i in range(len(x_batch))], y_batch
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        ybatches = self.dfy[index * self.batch_size:(index + 1) * self.batch_size]
        x,y = self.__get_data(batches, ybatches)
        print(np.asarray(x).shape)
        return [x, y]
    
    def on_poch_end(self):
        # if self.shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        pass