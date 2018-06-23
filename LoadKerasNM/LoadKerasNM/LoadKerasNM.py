from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils




(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test  = X_test.reshape(10000, 784)
X_test  = X_test.astype('float32')
y_test = np_utils.to_categorical(y_test, 10)



json_file = open('C:/Users/sergm/source/repos/KerasNN/KerasNN/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/sergm/source/repos/KerasNN/KerasNN/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
