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
import cv2
import imgaug

pytesseract.pytesseract.tesseract_cmd= "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
#Leer imágenes
#img = cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png")
#imgGris= cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png", 0)
BLACK= [0,0,0]
WHITE= [255,255,255]
plt.rcParams['figure.figsize'] = (7,7)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

X_train= []
Y_train= []
X_test =[]
Y_test= []
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
    size_box = list(box[1])
    center = box[0]
    box=list(box)
    print("entrando a cortar")
    

   # if(angle_box < 45):
        
        #aux1=box[1][0]
        #aux2=box[1][1] 
        #size_box[0]=aux1
        #size_box[1]=aux2

    angle_box += 90
    print(angle_box)
    if(angle_box == 90):
        angle_box= angle_box-90
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)
    
        cropped = cv2.getRectSubPix(input, (int(size_box[0]),int(size_box[1])), box[0])
        cropped = cv2.copyMakeBorder(cropped, 2,2,2,2,cv2.BORDER_CONSTANT, value=BLACK)
        return cropped
    else:
        transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
        input = cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]), cv2.INTER_CUBIC)
    
        cropped = cv2.getRectSubPix(input, (int(size_box[1]),int(size_box[0])), box[0])
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
        #imgCarta = RoInterestPruebas(imgCarta)
        binarizada = Binarization(imgCarta)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))


        MostrarImg(binarizada)
        eroded = cv2.erode(binarizada, kernel, 1)
        eroded = cv2.dilate(eroded, kernel, 2)
        dilated = cv2.dilate(eroded, kernel, iterations=10)
        #dilated = cv2.dilate(binarizada, kernel, iterations=15)
        MostrarImg(eroded)
        MostrarImg(dilated)
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
            cv2.drawContours(imgCarta,[box],0,(0,0,255),2)
            
            
            print(eroded.shape)
            ten_perc_h= int(eroded.shape[0]/20)
            ten_perc_l= int(eroded.shape[1]/20)
            eroded= eroded[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]
            
            
            MostrarImg(eroded)
            cropped = skewAndCrop(imgCarta, areas[i])
            text = identifyTextTesseract(eroded)
            print(text)
            MostrarImg(cropped)


            #text = identifyTextTesseract(imgCarta)
            #print(text)

            #MostrarImg(imgCarta)
            file.write(name)
            file.write("\n")
            file.write(text)
            file.write("\n")
        if (len(areas)==0):
            print(eroded.shape)
            ten_perc_h= int(eroded.shape[0]/20)
            ten_perc_l= int(eroded.shape[1]/20)
            eroded= eroded[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]
            
            
            MostrarImg(eroded)
            
            text = identifyTextTesseract(eroded)
            print(text)
            


def PrepararEntradas(input, name):
    print("----- PREPARANDO -----")
    #binarizada = Binarization(imgCarta)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))


    #MostrarImg(binarizada)
    eroded = cv2.erode(input, kernel, 1)
    eroded = cv2.dilate(eroded, kernel, 2)
    dilated = cv2.dilate(eroded, kernel, iterations=10)
    ten_perc_h= int(eroded.shape[0]/20)
    ten_perc_l= int(eroded.shape[1]/20)
    eroded= eroded[ten_perc_h:19*ten_perc_h, ten_perc_l:19*ten_perc_l]
    mid= int(eroded.shape[0]/2)  
    l1= eroded.shape[1]-1
    l= int(l1/2)-1
    cut_l1= int(11*eroded.shape[1]/30)
    cut_l2= int(14*eroded.shape[1]/30)
    first_line= eroded[1:mid, 1:cut_l1]
    second_line= eroded[mid:(mid*2)-2, 1:l]        
    elements=[]
    print(first_line.shape)
    print(second_line.shape)
    ll1= first_line.shape[1]
    ll2= int(second_line.shape[1]/30)
    l2_e1= second_line[1:l, 1:15*ll2-1]
    l2_e2= second_line[1:l, 16*ll2-1:30*ll2-1]
    l2_e3= second_line[1:l, 29*ll2-1:l-1]
    MostrarImg(first_line)
    cv2.imwrite('l2_e1.png',l2_e1)
    cv2.imwrite('l2_e2.png',l2_e2)
    cv2.imwrite('l2_e3.png',l2_e3)
    old_l2_e1 = Image.open('l2_e1.png')
    old_l2_e2 = Image.open('l2_e2.png')
    old_l2_e3 = Image.open('l2_e3.png')
    old_size1= l2_e1.shape
    old_size2= l2_e2.shape
    old_size3= l2_e3.shape
    print("-----")
    print(old_size1)
    new_size=(60,60)
   

    new_l2_e1 = cv2.copyMakeBorder(l2_e1, 9, 9,16, 16, cv2.BORDER_CONSTANT, value=WHITE)
    new_l2_e2 = cv2.copyMakeBorder(l2_e2, 9, 9,16, 16, cv2.BORDER_CONSTANT, value=WHITE)
    new_l2_e3 = cv2.copyMakeBorder(l2_e3, 9, 9,16, 16, cv2.BORDER_CONSTANT, value=WHITE)
    old_l2_e2 = Image.open('l2_e2.png')
    old_l2_e3 = Image.open('l2_e3.png')
    MostrarImg(new_l2_e1)
    MostrarImg(new_l2_e2)
    MostrarImg(new_l2_e3)
    elements.append(first_line)
    elements.append(new_l2_e1)
    elements.append(new_l2_e2)
    elements.append(new_l2_e3)
'''
        first_line = cv2.dilate(first_line, kernel, 2)
        MostrarImg(cv2.convertScaleAbs(first_line))
        cv2.imwrite('first_line.png', first_line)
        #cv2.imwrite('second_line.png', second_line)
        first_line=cv2.imread('first_line.png', cv2.CV_8UC1)
    
        retval, first_line= cv2.threshold(first_line, 0, 255, cv2.THRESH_OTSU)

        imgCount, cnts, h = cv2.findContours(first_line, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imwrite('first_line.png', first_line)
        #first_line = cv2.imread('first_line.png')
        areas = []

        print(len(cnts))
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
            cv2.drawContours(imgCarta,[box],0,(0,0,255),2)
            
            
            print(eroded.shape)
            MostrarImg(eroded)
            cropped = skewAndCrop(imgCarta, areas[i])
            MostrarImg(cropped)
        #dilated = cv2.dilate(binarizada, kernel, iterations=15)
        MostrarImg(first_line)
        MostrarImg(second_line)
        dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        input = np.asarray(input)
        input = input.astype(np.float)
        input /= 255
'''
   


def OCRDeepLearning(X):
        
   
    
   
    

    json_file = open('C:/Users/sergm/source/repos/KerasNN/KerasNN/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:/Users/sergm/source/repos/KerasNN/KerasNN/model.h5")
    print("Loaded model from disk")
    #check_pointer = callbacks.ModelCheckpoint("C:/Users/sergm/source/repos/KerasNN/KerasNN/model_checkpoint.h5", save_best_only=True)

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    answers = loaded_model.predict_classes(X)
    return answers
    


   


if __name__ == "__main__":
    for i in listdir("C:/Users/sergm/source/repos/OCRTry/Generadas/Samples"):
        print(i)
        name= "C:/Users/sergm/source/repos/OCRTry/Generadas/Samples/%s" % i
        print(name)
        #print(name)
        imgCarta= cv2.imread(name) 
        #img2 = cv2.imread("C:/Users/sergm/source/repos/OCRTry/Reales/carta_entera4.jpg")
        #imgCarta = img2
        #OCR(imgCarta, name)
        PrepararEntradas(imgCarta, name)
        OCRDeepLearning()
        
        



