import numpy as np
import keras
import matplotlib.image as mpimg

class Siamese_data_gen(keras.utils.Sequence):
    def __init__(self, paths, classes, batch_size, input_size):
        self.paths = paths
        self.classes = classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_classes = len(set(classes))
        self.class_indices = dict(zip(sorted(set(classes)), range(self.n_classes)))
        self.indexes = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for i in indexes:
            x1 = mpimg.imread(self.paths[i])
            c1 = self.class_indices[self.classes[i]]
            for j in range(i+1, len(self.paths)):
                x2 = mpimg.imread(self.paths[j])
                c2 = self.class_indices[self.classes[j]]
                if c1 == c2:
                    y = [1, 0]
                else:
                    y = [0, 1]
                x_batch.append([x1, x2])
                y_batch.append(y)
        x_batch = [np.array(x_batch)[:, :, :, :, i] for i in range(2) for _ in range(2)]
        y_batch = np.array(y_batch)
        return x_batch, y_batch
    
    def on_epoch_end(self):
        # if self.shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        print('epoch end')