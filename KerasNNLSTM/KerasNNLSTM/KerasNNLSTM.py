import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


data =  [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1, 101)]]
target = np.array(target, dtype=float)
