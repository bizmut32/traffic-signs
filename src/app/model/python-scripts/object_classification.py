import numpy as np
import random
from skimage.transform import SimilarityTransform, resize, warp, rotate, rescale
from skimage import data
import math
import time
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import Callback

def transform(img, translation=(0, 0), rotation=0.0, scale=1.0):
    translation = (translation[0] + img.shape[0] / 2 * (1 - scale), translation[1] + img.shape[1] / 2 * (1 - scale))
    transform = SimilarityTransform(translation=translation, scale=scale)
    rotated = rotate(img, rotation, mode='reflect')
    return warp(rotated, transform, mode='reflect')
    
def applyRandomTransformation(img):
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    dphi = random.uniform(-7, 7)
    dscale = random.uniform(0.8, 1.2)
    return transform(img, (dx, dy), dphi, dscale)

def transformImages(imgs):
    return np.array([applyRandomTransformation(img) for img in imgs]) 

class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=12):
        self.x, self.y = x_set, y_set
        self.orig_x, self.orig_y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        low, high = idx * self.batch_size, (idx + 1) * self.batch_size
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        print('Transforming images')
        start = time.time()
        self.x = transformImages(self.orig_x)
        end = time.time()
        print(int(end - start), 's')

class CNN:
    def __init__(self):
        self.model = self.getModel()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def getModel(self):
        model = Sequential()
        model.add(Conv2D(40, kernel_size=7, padding='same', activation='relu', input_shape=(40, 40, 3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(20, kernel_size=4, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(10, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dense(43, activation='softmax')) #hyperbolic tangent?
        return model
        
    def train(self, x, y, xValid, yValid):
        start = time.time()
        self.history = self.model.fit(
            Generator(x, y), epochs = 12,
            validation_data = (xValid, yValid),
            verbose=2)
        
        end = time.time()
        print('Training model took', int(end - start), 's')
    def evaluate(self, x, y):
        print('Accuracy:', self.model.evaluate(x, y, verbose=0)[1])
    
    def load(self, model):
        self.model = model

class MLP:
    def __init__(self):
        self.model = self.getModel()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def setModel(self, model):
        self.model = model
        

    def getModel(self):
        model = Sequential()
        model.add(Dense(1000, activation='relu', input_dim=5000))
        model.add(Dropout(0.33))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.33))
        model.add(Dense(43, activation='softmax'))
        return model
        
    def train(self, x, y, xValid, yValid):
        start = time.time()
        self.history = model.fit(x, y, epochs = 20,
            validation_data = (xValid, yValid),
            verbose=2)
        
        end = time.time()
        print('Training model took', int(end - start), 's')
    def evaluate(self, x, y):
        print('Accuracy:', self.model.evaluate(x, y, verbose=0)[1])
    
    def load(self, model):
        self.model = model

class Trainer():
    def __init__(self, dataSets):
        self.dataSets = dataSets
        self.models = [ CNN() for _ in dataSets ]
    def train(self):
        start = time.time()
        for i in range(0, len(self.dataSets)):
            print('Training model', i)
            self.models[i].train(self.dataSets[i][0], self.dataSets[i][1], 
                                 self.dataSets[i][2], self.dataSets[i][3])

        end = time.time()
        print('The whole process took', int(end - start), 's')
    def getCommittee(self):
        cCNN = CommitteeOfCNNs()
        for cnn in self.models:
            cCNN.addModel(cnn)
        return cCNN

class CommitteeOfCNNs():
    def __init__(self):
        self.models = []
        
    def addModel(self, model):
        self.models.append(model)
    
    def evaluate(self, dataSets):
        for i in range(0, len(dataSets)):
            self.models[i].evaluate(dataSets[i][4], dataSets[i][5])
    
    def predict(self, x):
        results = np.array([ self.models[i].model.predict(np.array([x[i]]))[0] for i in range(0, len(self.models)) ])
        votes = np.argmax(results, axis=1)
        winner = np.argmax(np.bincount(votes, minlength=43))
        return winner

    def test(self, x, y):
        x = self.transpose(x)
        y = np.argmax(y, axis=1)
        predictions = np.array([ self.predict(row) for row in x ])
        success = predictions == y
        accuracy = len(y[success]) / len(y)
        return accuracy
    
    def transpose(self, x):
        X = []
        for i in range(0, len(x[0])):
            X.append([None, None, None, None, None])
            
        for i in range(0, len(x)):
            for j in range(0, len(x[i])):
                X[j][i] = x[i][j]
        return X
    
    def save(self):
        for i in range(0, len(self.models)):
            self.models[i].model.save('models/model_committee%x.h5' % i)
    
    def load(self, path="models"):
        self.models = []
        for i in range(0, 5):
            model = load_model('%s/model_committee%x.h5'%(path, i))
            cnn = CNN()
            cnn.load(model)
            self.addModel(cnn)