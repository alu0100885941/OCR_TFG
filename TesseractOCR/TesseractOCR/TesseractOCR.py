#Operaciones básicas con imágenes.

import cv2
import sys
import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from pytesseract import image_to_string
import string
import pytesseract
import skimage
from TFANN import ANNC

pytesseract.pytesseract.tesseract_cmd= "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
#Leer imágenes

imgCarta= cv2.imread("C:/Users/sergm/source/repos/OCRTry/carta_entera5.jpg") 

#Architecture of the neural network
NC = len(string.ascii_letters + string.digits + ' ')
MAX_CHAR = 64
IS = (14, 640, 3)       #Image size for CNN
ws = [('C', [4, 4,  3, NC // 2], [1, 2, 2, 1]), ('AF', 'relu'), 
      ('C', [4, 4, NC // 2, NC], [1, 2, 1, 1]), ('AF', 'relu'), 
      ('C', [8, 5, NC, NC], [1, 8, 5, 1]), ('AF', 'relu'),
      ('R', [-1, 64, NC])]

#Reconocimiento con Tesseract


def identifyTextTesseract(input):
    
    text = image_to_string(input, "eng")
    return text

   
#Generate datasheet to train the network 
def MakeImg(t, f, fn, s = (100, 100), o = (16, 8)):
    '''
    Generate an image of text
    t:      The text to display in the image
    f:      The font to use
    fn:     The file name
    s:      The image size
    o:      The offest of the text in the image
    '''
    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(OFS, t, (255, 255, 255), font = f)
    img.save(fn)


def LoadData(FP = '.'):
    '''
    Loads the OCR dataset. A is matrix of images (NIMG, Height, Width, Channel).
    Y is matrix of characters (NIMG, MAX_CHAR)
    FP:     Path to OCR data folder
    return: Data Matrix, Target Matrix, Target Strings
    '''
    TFP = os.path.join(FP, 'Train.csv')
    A, Y, T, FN = [], [], [], []
    with open(TFP) as F:
        for Li in F:
            FNi, Yi = Li.strip().split(',')                     #filename,string
            T.append(Yi)
            A.append(cv2.imread(os.path.join(FP, 'Out', FNi)))
            Y.append(list(Yi) + [' '] * (MAX_CHAR - len(Yi)))   #Pad strings with spaces
            FN.append(FNi)
    return np.stack(A), np.stack(Y), np.stack(T), np.stack(FN)

A, Y, T, FN = LoadData()

#Create the neural network in TensorFlow
cnnc = ANNC(IS, ws, batchSize = 64, learnRate = 5e-5, maxIter = 32, reg = 1e-5, tol = 1e-2, verbose = True)
#Fit the network
cnnc.fit(A, Y)
#The predictions as sequences of character indices
YH = np.zeros((Y.shape[0], Y.shape[1]), dtype = np.int)
for i in np.array_split(np.arange(A.shape[0]), 32): 
    YH[i] = np.argmax(cnnc.predict(A[i]), axis = 2)
#Convert from sequence of char indices to strings
PS = [''.join(CS[j] for j in YHi) for YHi in YH]
for PSi, Ti in zip(PS, T):
    print(Ti + '\t->\t' + PSi)



'''
    #The possible characters to use
    CS = list(string.ascii_letters) + list(string.digits)
    RTS = list(np.random.randint(10, 64, size = 8192)) + [64]
    #The random strings
    S = [''.join(np.random.choice(CS, i)) for i in RTS]
    #Get the font
    font = ImageFont.truetype("C:/Users/sergm/source/repos/OCRTry/OCRTry/Aaargh.ttf", 16)
    #The largest size needed
    MS = max(font.getsize(Si) for Si in S)
    #Computed offset
    OFS = ((640 - MS[0]) // 2, (32 - MS[1]) // 2)
    #Image size
    MS = (640, 32)
    Y = []
    for i, Si in enumerate(S):
        MakeImg(Si, font, str(i) + '.png', MS, OFS)
        Y.append(str(i) + '.png,' + Si)
    #Write CSV file
    with open('Train.csv', 'w') as F:
        F.write('\n'.join(Y))



        text = identifyTextTesseract(imgCarta)
        print(text)


        plt.imshow(imgCarta)

        plt.show()
'''



