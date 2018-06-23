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
import pytesseract
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
pytesseract.pytesseract.tesseract_cmd= "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
#Leer imágenes
#img = cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png")
#imgGris= cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png", 0)
BLACK= [0,0,0]
#api = tesseract.TessBaseApi()
#Escribir Imágenes
'''
    cv2.imwrite("C:/Users/sergm/source/repos/OCRTry/pruebaGris.png", imgGris)

    #img[1:200]= [255,255,255]

    cv2.imwrite("C:/Users/sergm/source/repos/OCRTry/pruebaMachaque.png", img)
    #Propiedades de imágen
    print(img.shape)
    magicTG = cv2.imread("C:/Users/sergm/source/repos/OCRTry/rules_lawyer_judge_promo_january_2018.jpg")
    #Región de Imagen
    title = magicTG[1:100, 1:magicTG.shape[1]-40]
    cv2.imwrite("C:/Users/sergm/source/repos/OCRTry/pruebaRecorte.png", title)
    #Cambiar colores a gusto de posiciones
    img[:,:,1]=140
    cv2.imwrite("C:/Users/sergm/source/repos/OCRTry/pruebaModificada.png", img)
'''

#Trabajar con vectores 
'''
    imgGris.itemset((10,10,2),100)

    x = np.random.randint(9, size=(3, 3))
    print(x)
    x.itemset(4,0)
    print("-----------------")
    print(x)
    x.itemset((2,2), 9)
    print("-----------------")
    print(x)
'''



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
    size_box = box[1]
    '''
        if(angle_box < -45):
            angle_box += 90
            aux1=box[1][0]
            aux2=box[1][1] 
            box[1][1]=aux1
            box[1][0]=aux2
    '''
    #transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
    #rotated = cv2.warpAffine(input, transform, (input.shape[0], input.shape[1]), cv2.INTER_CUBIC)
    
    cropped = cv2.getRectSubPix(input, (int(size_box[0]),int(size_box[1])), box[0])
    cropped = cv2.copyMakeBorder(cropped, 2,2,2,2,cv2.BORDER_CONSTANT, value=BLACK)
    return cropped

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
    h1= input.shape[0]-1 
    h= int(4*(h1/10))
    #print(h1)
    l1= input.shape[1]-1
    l= int(l1/3)-1
    #print(l1)
    output= input[1:h, 1:l]
    #MostrarImg(output)
    mid= int(output.shape[0]/2)
    cut_l1= int(11*output.shape[1]/30)
    cut_l2= int(14*output.shape[1]/30)
    first_line= output[1:mid, 1:l]
    second_line= output[mid:(mid*2)-2, 1:l]
    #MostrarImg(first_line)
    #MostrarImg(second_line)
    #return output, first_line, second_line
    return output

def OCR(imgCarta, name):
        file = open("output.txt", "a")
        #imgCarta = RoInterestCartasReales(imgCarta)
        imgCarta = RoInterestPruebas(imgCarta)
        binarizada = Binarization(imgCarta)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))


        #MostrarImg(binarizada)

        dilated = cv2.dilate(binarizada, kernel, iterations=7)
        #dilated = cv2.dilate(binarizada, kernel, iterations=15)
        #MostrarImg(dilated)
        dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        imgCount, cnts, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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
            cv2.drawContours(imgCarta,[box],0,(0,0,255),3)

            cropped = skewAndCrop(imgCarta, areas[i])
            text = identifyTextTesseract(cropped)
            #print(text)
            #MostrarImg(cropped)


            text = identifyTextTesseract(imgCarta)
            print(text)

            #MostrarImg(imgCarta)
            file.write(name)
            file.write("\n")
            file.write(text)
            file.write("\n")


def OCRDeepLearning():
        

        nb_classes = 10
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print(len(X_train[0]))
        
        print("X_train original shape", X_train.shape)
        print("y_train original shape", y_train.shape)

  

        X_train = X_train.reshape(60000, 784)
        X_test  = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')
        X_train /= 255
        X_test  /= 255
        print("Trainning matrix shape: ", X_train.shape)
        print("Testing matrix shape: ", X_test.shape)

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        model = Sequential()
        model.add(Dense(512, input_dim=784))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.compile(loss=keras.losses.categoricarl_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        #model.fit(X_train, y_train, batch_size=128, epoch=4, show_acuracy=True, verbose=1, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, 128, 4, 1, validation_data=(X_test, y_test))

        score = model.evaluate(X_test, y_test, verbose=0)
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

if __name__ == "__main__":
    for i in range(0,1000):
        name= "C:/Users/sergm/source/repos/OCRTry/Generadas/prueba_salida%d.png" % i
        print(name)
        imgCarta= cv2.imread(name) 
        #img2 = cv2.imread("C:/Users/sergm/source/repos/OCRTry/Reales/carta_entera4.jpg")
        #imgCarta = img2
        #OCR(imgCarta, name)
        OCRDeepLearning()



