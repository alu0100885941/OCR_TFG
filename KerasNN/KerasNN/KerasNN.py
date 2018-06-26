import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)
import keras
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, LSTM, Flatten, TimeDistributed, Reshape
from os import listdir
from os import walk
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D
from keras import backend as K
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.layers import TimeDistributed
import cv2
import string

letras = list(string.ascii_uppercase)
numbers = list(range(0-9))
dictionary = letras.append(numbers)
nb_classes = 10

X_train= []
Y_train= []
X_test =[]
Y_test= []
nb_classes = 37
for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    img =cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/%s"%file)
    X_train.append(img)
for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    file=open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/%s"%file)
    str=file.read
    Y_train.append(str)

for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    img =cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/%s"%file)
    X_test.append(img)
for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    file=open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers/%s"%file)
    str=file.read
    Y_test.append(str)




img =cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/test1.png")
plt.imshow(X_train[0])
plt.show()
cv2.waitKey(0)
#print(X_train[0].shape)
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_test)

'''
    #print("X_train original shape", X_train.shape)
    #print("y_train original shape", Y_train.shape)

        for i in range(9):
            plt.subplot(3,3, i+1)
            plt.imshow(X_train[i], cmap= 'gray',interpolation='none')
            plt.title("Class {}".format(y_train[i]))
 
    #print(X_train.shape)
    #X_train = X_train.reshape(1000, X_train[0].shape[0], X_train[0].shape[1], 1)
    #X_test  = X_test.reshape(1000, X_train[0].shape[0], X_train[0].shape[1], 1)
    #X_train = X_train.astype('float32')
    #X_test  = X_test.astype('float32')
    #X_train /= 255
    #X_test  /= 255
    #print("Trainning matrix shape: ", X_train.shape)
    #print("Testing matrix shape: ", X_test.shape)
    
    #print(X_train[0].shape[0])
    #print(X_train[0].shape[1])
'''


print(X_train.shape)

#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(Conv2D(256, (3,3), activation="relu", input_shape=(X_train[0].shape[0], X_train[0].shape[1], 3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(512, (5,5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Reshape((16, 12, 128)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(512,return_sequences=True))
model.add(Flatten())
#model.add(Reshape((37,-1)))
model.add(Dense(37, activation="softmax"))
model.summary()
'''
    #model.add(Dense(512, input_dim=784))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(10))
    #model.add(Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, verbose=1, validation_data=(X_test, Y_test))
#model.fit(X_train, Y_train, 128, 4, 1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test acuracy: ', score[1])
'''
            print('Test score:', score[0])
            print('Test acuracy: ', score[1])

            predicted_classes = model.predict_classes(X_test)
            correct_indices = np.nonzero(predicted_classes == y_test)[0]
            incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
            plt.figure()
            for i, correct in enumerate(correct_indices[:9]):
                plt.subplot(3,3,i+1)
                plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
                plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
            plt.figure()
            for i, incorrect in enumerate(incorrect_indices[:9]):
                plt.subplot(3,3,i+1)
                plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
                plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)

            model.save_weights("model.h5")
            print("Saved model to disk")
'''
