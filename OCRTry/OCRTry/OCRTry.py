#Operaciones básicas con imágenes.
from os import listdir
from os import walk
import cv2
import sys
import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from pytesseract import image_to_string
import pytesseract
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import imgaug
import keras
import tensorflow as tf
from os.path import join
import json
import random
import itertools
import re
import datetime
#import cairocffi as cairo
#import editdistance
from keras.models import model_from_json
from scipy import ndimage
import pylab
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from scipy.misc import imsave
from tkinter.filedialog import askdirectory

from collections import Counter
pytesseract.pytesseract.tesseract_cmd= "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

BLACK= [0,0,0]
WHITE= [255,255,255]
plt.rcParams['figure.figsize'] = (7,7)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

X_train= []
Y_train= []
X_test =[]
Y_test= []



#Threshold  y binarización de la imagen. Preparar OCR
def MostrarImg(input):
    plt.imshow(input)
    plt.show()
    cv2.waitKey(0)

def Binarization(input):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    binary_image = input
    retval, binary_image= cv2.threshold(input, 0, 255, cv2.THRESH_OTSU)
    
    cv2.imwrite('Binary.png', binary_image)
    binary_image = Image.open("Binary.png")
    clrs = binary_image.getcolors()
    #print(clrs)
    #n_white = cv2.countNonZero(binary_image)
    #n_black = binary_image.size().area() - n_white
    
    if(clrs[0][0] > clrs[1][0]):
     #print("Imagen normal")
     binary_image= cv2.imread("Binary.png")
     
     return binary_image
     
    else:
     #print("Imagen invertida")
     
     binary_image= cv2.imread("Binary.png")
     binary_image = cv2.bitwise_not(binary_image)
    
     return  binary_image

def skewAndCrop(input, box):
    angle_box = box[2]
    size_box = list(box[1])
    center = box[0]
    box=list(box)
    #print("entrando a cortar")
    

   # if(angle_box < 45):
        
        #aux1=box[1][0]
        #aux2=box[1][1] 
        #size_box[0]=aux1
        #size_box[1]=aux2

    #angle_box += 90
    #print(angle_box)
    if(angle_box > -45):
        angle_box= angle_box
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)
    
        cropped = cv2.getRectSubPix(input, (int(size_box[0]),int(size_box[1])), box[0])
        #cropped = cv2.copyMakeBorder(cropped, 2,2,2,2,cv2.BORDER_CONSTANT, value=BLACK)
        return cropped
    else:
        angle_box += 90
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)
    
        cropped = cv2.getRectSubPix(input, (int(size_box[1]),int(size_box[0])), box[0])
        #cropped = cv2.copyMakeBorder(cropped, 2,2,2,2,cv2.BORDER_CONSTANT, value=BLACK)
        return cropped
   
def skewAndCropWithCords(input, box):
    angle_box = box[2]
    size_box = list(box[1])
    center = box[0]
    box=list(box)
    data=[]
    if(angle_box > -45):
        angle_box= angle_box
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)    
        cropped = cv2.getRectSubPix(input, (int(size_box[0]),int(size_box[1])), box[0])
        data.append(cropped)
        data.append(box[0])
        return data
    else:
        angle_box += 90
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)
        cropped = cv2.getRectSubPix(input, (int(size_box[1]),int(size_box[0])), box[0])
        data.append(cropped)
        data.append(box[0])
        return data
    

def identifyTextTesseract(input):
    
    text = image_to_string(input, "eng")
    return text

def RoInterestCartasReales(input):
    h1= input.shape[0]-1 
    h= int(10*(h1/11))
    #print(h1)
    l1= input.shape[1]-1
    l= int(l1/4)-1
    #print(l1)
    output= input[h:h1, 1:l]
    #MostrarImg(output)
    return output




def RoInterestPruebas(input):
    ten_perc_h= int(input.shape[0]/20)
    ten_perc_l= int(input.shape[1]/20)
    output= input[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]

    return output

def OCR(imgCarta, name):
        print("OCR-COMPUTER VISION")
        file = open("output.txt", "a")
        imgCarta = RoInterestPruebas(imgCarta)
        binarizada = Binarization(imgCarta)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        dilated = cv2.cvtColor(binarizada, cv2.COLOR_BGR2GRAY)   
        imgCount, cnts, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
        areas = []


        for i in range(len(cnts)):
            box = cv2.minAreaRect(cnts[i])
    
            if(box[1][0] < 20 or box[1][1] < 20):
                continue
            angle_box = box[2]
            if(angle_box < -45.0):
                proportion = box[1][0] / box[1][1]
            else:
                proportion = box[1][1] /  box[1][0]
            if(proportion > 0.5):
                continue
            areas.append(box)
 


 
  
        for i in range(len(areas)):
            box= cv2.boxPoints(areas[i])
            box= np.int0(box)
            print(binarizada.shape)
            ten_perc_h= int(binarizada.shape[0]/20)
            ten_perc_l= int(binarizada.shape[1]/20)
            eroded= binarizada[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]
            cropped = skewAndCrop(binarizada, areas[i])
            text = identifyTextTesseract(binarizada)
            print(text)
            MostrarImg(eroded)

        if (len(areas)==0):
            text = identifyTextTesseract(imgCarta)
            
        text = identifyTextTesseract(imgCarta)
        
        return text
           
            
def sec_elem(s):
    
    return s[1][1] 
def fst_elem(s):
    
    return s[1][0] 

def PrepararEntradas(input, name):
    print("----- PREPARANDO -----")
    ten_perc_h= int(input.shape[0]/20)
    ten_perc_l= int(input.shape[1]/20)
    input= input[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    bw= Binarization(input)
    eroded= bw
    copy= eroded
    eroded = input      
    elements=[]
    cv2.imwrite('bw.png', copy)
    id_tot=cv2.imread('bw.png', cv2.CV_8UC1)
    retval, id_tot= cv2.threshold(id_tot, 0, 255, cv2.THRESH_OTSU)
    imgCount, cnts, h = cv2.findContours(id_tot, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
    new_size=(60,60)
    areas = []




 
    for i in range(len(cnts)):
        box = cv2.minAreaRect(cnts[i])
    
        if(box[1][0] < 5 or box[1][1] < 5):
            continue
        angle_box = box[2]
        if(angle_box < -45.0):
            proportion = box[1][0] / box[1][1]
        else:
            proportion = box[1][1] /  box[1][0]

        areas.append(box)

    for i in range(len(areas)):
        box= cv2.boxPoints(areas[i])
        box= np.int0(box)
        cropped = skewAndCropWithCords(input, areas[i])
        actual_x= cropped[0].shape[1]
        actual_y= cropped[0].shape[0]
        rest_x= 60-actual_x
        rest_y= 60-actual_y
        if(rest_y%2 ==0):
            rest_y=int(rest_y/2)
            if(rest_x%2 ==0):
                rest_x=int(rest_x/2)
                cropped[0] = cv2.copyMakeBorder(cropped[0], rest_y, rest_y,rest_x, rest_x, cv2.BORDER_CONSTANT, value=WHITE)
            else:
                rest_x=rest_x-1
                rest_x=int(rest_x/2)
                cropped[0] = cv2.copyMakeBorder(cropped[0], rest_y, rest_y,rest_x, rest_x+1, cv2.BORDER_CONSTANT, value=WHITE)
        else:
            rest_y=rest_y-1
            rest_y=int(rest_y/2)
            if(rest_x%2 ==0):
                rest_x=int(rest_x/2)
                cropped[0] = cv2.copyMakeBorder(cropped[0], rest_y+1, rest_y,rest_x, rest_x, cv2.BORDER_CONSTANT, value=WHITE)
            else:
                rest_x=rest_x-1
                rest_x=int(rest_x/2)
                cropped[0] = cv2.copyMakeBorder(cropped[0], rest_y+1, rest_y,rest_x, rest_x+1, cv2.BORDER_CONSTANT, value=WHITE)

        elements.append(cropped)
    up_elem=[]
    down_elem=[]
    print("-------- FIN ---------")
    for imgE in elements:
        if (sec_elem(imgE) < input.shape[0]/2):
            up_elem.append(imgE)
            
        else:
            down_elem.append(imgE)

    up_elem= sorted(up_elem, key=fst_elem)
    down_elem= sorted(down_elem, key=fst_elem) 
    return([up_elem[0:3], down_elem[0:3]])
  


      

    


   


def OCRDeepLearning(X):
        
    X_up = []
    X_down = []
    for element in X[0]:
        X_up.append(element[0])
        
    for element in X[1]:
        X_down.append(element[0])
        

    print("OCR-DEEPLEARNING")
    X_up = np.asarray(X_up)
    X_down = np.asarray(X_down)
    X_up = X_up.astype(np.float)
    X_down = X_down.astype(np.float)
    X_up /= 255
    X_down /= 255
    json_file = open('C:/Users/sergm/source/repos/KerasNN/KerasNN/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:/Users/sergm/source/repos/KerasNN/KerasNN/model2_vf.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    answers_up = loaded_model.predict_classes(X_up)
    answers_down = loaded_model.predict_classes(X_down)
    print("Primera linea: ", answers_up)
    print("Segunda linea: ",answers_down)
    cut=0
    str_answ=[]
    char=""
    for ans in answers_up:
        ans = "%d"%ans
        str_answ.append((ans))
    for ans in answers_down:
         str_answ.append(chr(ans+55))

    return  str_answ
    

def OCR_CartasReales(input):
     OCR(img2, name)

   


if __name__ == "__main__":
    cont2=0
    folder = askdirectory()
    for i in listdir(folder):
        cont=0
        name=  folder+"/"+ i
        lname= "%s"% i
        imgCarta= cv2.imread(name) 
        MostrarImg(imgCarta)
        img2 = cv2.imread("C:/Users/sergm/source/repos/OCRTry/Reales/carta_entera4.jpg")
        img2 = RoInterestCartasReales(img2)
        name= "carta_entera4.jpg"
        text= OCR(img2, name)
        
        print("------ RESULTADO -----")
        print(text)
        MostrarImg(imgCarta)
        result = PrepararEntradas(imgCarta, lname)
        str_answ= OCRDeepLearning(result)
        print("Resultado: ",str_answ)
        print("Cadena: ","".join(str_answ))
        MostrarImg(imgCarta)
       
        
 

        



