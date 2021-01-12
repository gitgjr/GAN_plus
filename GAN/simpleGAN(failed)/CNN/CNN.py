import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def setdata():
    (train_X,train_Y),(test_X,test_Y) = keras.datasets.mnist.load_data()
    print(train_X.shape,train_Y.shape,test_X.shape)
    train_X = train_X.reshape(60000,28,28,1)
    test_X  = test_X.reshape(10000,28,28,1)
    train_X = train_X/255.0
    test_X = test_X/255.0
    return train_X,train_Y,test_X,test_Y
class CNN(object):
    def __init__(self):
        network = models.Sequential()
        network.add(layers.Conv2D(32,(3*3),activation='relu',input_shape=(28,28,1)))
        network.add(layers.MaxPool2D(pool_size=(2,2)))
        network.add(layers.Conv2D(64,(3,3),activation='relu'))
        network.add(layers.MaxPool2D(pool_size=(2,2)))
        network.add(layers.Conv2D(64, (3, 3), activation='relu'))

        network.add(layers.Flatten())
        network.add(layers.Dense(64,activation='relu'))
        network.add(layers.Dense(10,activation='softmax'))

        # network.summary()
        self.network = network

class Train():
    def __init__(self):
        self.CNN = CNN()
        

    def train(self):
        train_X, train_Y, test_X, test_Y = setdata()
        check_path ='./ckpt/cp-{epoch:04d}.ckpt'
        save_model = keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,save_freq=5)
        self.CNN.network.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        self.CNN.network.fit(train_X,train_Y,epochs=5,callbacks=[save_model])#
        test_loss,test_acc = self.CNN.network.evaluate(test_X,test_Y)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(test_Y)))
#
# app = Train()
# app.train()