# need to ref og code

import numpy as np
#import keras
from tensorflow import keras

class DataGen(keras.utils.Sequence):
    
    # OG: not sure if we need all these vars
    #def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
    #            n_classes=10, shuffle=True):
    def __init__(self, list_IDs, batch_size=32, dim=29351, n_channels=1, n_classes=10,
            shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        x = np.empty((self.batch_size, self.dim))
        # need to modify y here to be same size as x
        y = np.empty((self.batch_size, self.dim), dtype=int)

        for idx, identifier in enumerate(list_IDs_temp):
            x[idx,] = np.load(f"data/{identifier}_x.npy")
            # get y here as well
            y[idx,] = np.load(f"data/{identifier}_y.npy")

        # they return y as keras.utils.to_categorical(y, num_classes = self.n_classes)
        return x, y
