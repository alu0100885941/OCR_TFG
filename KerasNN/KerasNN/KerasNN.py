from keras.layers import Convolution2D, MaxPooling2D, Dropout, LSTM, Flatten, TimeDistributed, Reshape
from os import listdir
from os import walk
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D
from keras import backend as K
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras import callbacks
from numpy import argmax
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import cv2
import string
import unicodedata
import pydot
import array


plt.rcParams['figure.figsize'] = (7,7)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
letras = list(string.ascii_uppercase)
numbers = list(range(0-9))
dictionary = letras.append(numbers)
X_train= []
Y_train= []
X_test =[]
Y_test= []
counter=0
nb_classes = 37
#for letra in letras:
    #print(letra, ord(letra)-55)
list_comprob=[]
print("----- PREPARANDO -----")
#for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Modified"):
for file in listdir("C:/Users/sergm/source/repos/OCRTry/Elementos/Train/Samples"):
    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    

    #img =cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Modified/%s"%file)
    img =cv2.imread("C:/Users/sergm/source/repos/OCRTry/Elementos/Train/Samples/%s"%file)
    #zeros = np.zeros((60,60, 3),float)
    #zeros[:img.shape[0],:img.shape[1]]=img
    X_train.append(img)
    
#for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers"):
for file in listdir("C:/Users/sergm/source/repos/OCRTry/Elementos/Train/Answers"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    #fich=open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/%s"%file)
    
    file=open("C:/Users/sergm/source/repos/OCRTry/Elementos/Train/Answers/%s"%file)
    str=file.read()
    list_comprob.append(str)
    #print("%s"%file.name, str)
    #print(str, len(str))
    
    if(str != "\n" ):
        if(str.isnumeric()):
            Y_train.append(str)
            counter=counter+1
        
        else:
            aux= ord(str)-55
            if(aux<37):
                Y_train.append(aux)
                counter=counter+1
            else:
                 new_X_train= X_train[0:counter-1]
                 new_X_train2= X_train[counter:len(X_train)]
                 X_train = new_X_train+new_X_train2
                 counter=counter-1
    else:
         print("detectamos vacio")
         print(counter)
         print(len(X_train[1]))
         new_X_train= X_train[0:counter-1]
         new_X_train2= X_train[counter:len(X_train)]
         X_train = new_X_train+new_X_train2
         counter=counter-1

        
second_l= list(set(list_comprob))
print(second_l)
        

counter=0


#for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Modified"):
for file in listdir("C:/Users/sergm/source/repos/OCRTry/Elementos/Test/Samples"):

    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    
    img =cv2.imread("C:/Users/sergm/source/repos/OCRTry/Elementos/Test/Samples/%s"%file)
    #zeros = np.zeros((60,60, 3),float)
    #zeros[:img.shape[0],:img.shape[1]]=img
    X_test.append(img)

print(len(X_test))
#for file in listdir("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers"):
for file in listdir("C:/Users/sergm/source/repos/OCRTry/Elementos/Test/Answers"):
    #print(len(X_test))
    #print("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/%s"%file)
    file=open("C:/Users/sergm/source/repos/OCRTry/Elementos/Test/Answers/%s"%file)
    str=file.read()
    #print(str, len(str))
    
    if(str != "\n" ):
        if(str.isnumeric()):
            Y_test.append(str)
            counter=counter+1
        
        else:
            aux= ord(str)-55
            if(aux<37):
                Y_test.append(aux)
                counter=counter+1
            else:
                 new_X_test= X_test[0:counter-1]
                 new_X_test2= X_test[counter:len(X_test)]
                 X_test = new_X_test+new_X_test2
                 counter=counter-1
    else:
         print("detectamos vacio")
         #print(counter)
         #print(len(X_test[1]))
         new_X_test= X_test[0:counter-1]
         new_X_test2= X_test[counter:len(X_test)]
         X_test = new_X_test+new_X_test2
         counter=counter-1
         #X_test= np.delete(X_test, counter)
       
        

#print(len(X_train), Y_train)
conta=0
print(len(X_test))

print("---------------------")
#print(X_train[0].shape)

X_train = np.asarray(X_train)
X_train = X_train.astype(np.float)
X_train /= 255

#X_train = X_train.reshape(8000, 90, 30, 1)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
X_test = X_test.astype(np.float)
X_test /= 255 
#X_test = X_test.reshape(2000,90,30,1)
Y_test = np.asarray(Y_test)


Y_train = np_utils.to_categorical(Y_train, num_classes=37)
Y_test = np_utils.to_categorical(Y_test, num_classes=37)
#img =cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Modified/test1.png")



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


'''

    #Y_train = np_utils.to_categorical(Y_train, nb_classes)
    #Y_test = np_utils.to_categorical(Y_test, nb_classes)
    #X_train[0].shape[0], X_train[0].shape[1]
    model = Sequential()
    model.add(Conv2D(512, (3,3), activation="relu", input_shape=(60,60,3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(512, (5,5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Reshape((16, 12, 128)))
    #model.add(TimeDistributed(Flatten()))
    #model.add(LSTM(512,return_sequences=True))
    #model.add(Flatten())
    #model.add(Reshape((37,-1)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(37, activation='softmax'))

    model.summary()
    #adam= keras.optimizers.Adam()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

'''
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




'''
json_file = open('C:/Users/sergm/source/repos/KerasNN/KerasNN/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/sergm/source/repos/KerasNN/KerasNN/model.h5")
print("Loaded model from disk")
#check_pointer = callbacks.ModelCheckpoint("C:/Users/sergm/source/repos/KerasNN/KerasNN/model_checkpoint.h5", save_best_only=True)

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
answers = loaded_model.predict_classes(X_train[0:10])
cut=0
str_answ=[]
char=""
for i in Y_train:
 if(argmax(i)>9):
     print(chr(argmax(i)+55))
 else:
     print(argmax(i))
 cut=cut+1
 if cut>9:
     break

for ans in answers:
    if(ans>9):
     str_answ.append(chr(ans+55))
     
    else:
     ans = "%d"%ans
     str_answ.append((ans))


print(str_answ)
print("".join(str_answ))
'''
#predictions = loaded_model.predict(X_train[0:10])
#rounded = [round(x[0]) for x in predictions]
#print(predictions)

print(len(Y_train))
json_file = open('C:/Users/sergm/source/repos/KerasNN/KerasNN/model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/sergm/source/repos/KerasNN/KerasNN/model2_vf.h5")
print("Loaded model from disk")
#check_pointer = callbacks.ModelCheckpoint("C:/Users/sergm/source/repos/KerasNN/KerasNN/model_checkpoint.h5", save_best_only=True)

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.fit(X_train, Y_train, epochs=1, verbose=1, validation_data=(X_test, Y_test))
#model.fit(X_train, Y_train, 128, 4, 1, validation_data=(X_test, Y_test))

score = loaded_model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test acuracy: ', score[1])

model_json = loaded_model.to_json()
with open("model2.json", "w") as json_file:
        json_file.write(model_json)

loaded_model.save_weights("model2_vf.h5")
print("Saved model to disk")


'''
            print('Test score:', score[0])
            print('Test acuracy: ', score[1])

            predicted_classes = model.predict_classes(X_test)
            correct_indices = np.nonzero(predicted_classes == Y_test)[0]
            incorrect_indices = np.nonzero(predicted_classes != Y_test)[0]
            plt.figure()
            for i, correct in enumerate(correct_indices[:9]):
                plt.subplot(3,3,i+1)
                plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
                plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Y_test[correct]))
    
            plt.figure()
            for i, incorrect in enumerate(incorrect_indices[:9]):
                plt.subplot(3,3,i+1)
                plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
                plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Y_test[incorrect]))

            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)

            model.save_weights("model.h5")
            print("Saved model to disk")
'''
