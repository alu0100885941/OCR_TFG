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

pytesseract.pytesseract.tesseract_cmd= "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
#Leer imágenes
img = cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png")
imgGris= cv2.imread("C:/Users/sergm/source/repos/OCRTry/prueba.png", 0)
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

def Binarization(input):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    binary_image = input
    retval, binary_image= cv2.threshold(input, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('Captura-Grayscale.png', binary_image)
    binary_image = Image.open("Captura-Grayscale.png")
    clrs = binary_image.getcolors()
    #n_white = cv2.countNonZero(binary_image)
    #n_black = binary_image.size().area() - n_white
   
    if(clrs[0][1] > clrs[1][1]):
     print("Imagen normal")
     binary_image= cv2.imread("Captura-Grayscale.png")
     return binary_image
     
    else:
     print("Imagen invertida")
     binary_image= cv2.imread("Captura-Grayscale.png")
     binary_image= cv2.bitwise_not(binary_image)
     return  binary_image

def skewAndCrop(input, box):
    angle_box = box[2]
    size_box = box[1]
    if(angle_box < -45):
        angle_box += 90
        box[1][0], box[1][1] = box[1][1], box[1][0]
    transform = cv2.getRotationMatrix2D(box[0], angle_box, 1.0)
    rotated = cv2.warpAffine(input, transform, (input.shape[0], input.shape[1]), cv2.INTER_CUBIC)
    
    cropped = cv2.getRectSubPix(rotated, (int(size_box[0]),int(size_box[1])), box[0])
    cropped = cv2.copyMakeBorder(cropped, 10,10,10,10,cv2.BORDER_CONSTANT, value=BLACK)
    return cropped

def identifyTextTesseract(input):
    
    text = image_to_string(input, "por")
    return text


def processTextRecogn(input):
    
    filtC1= cv2.text.loadClassifierNM1("trained_classifierNM1.xml")
    filt1= cv2.text.createERFilterNM1(filtC1)
    filtC2= cv2.text.loadClassifierNM2("trained_classifierNM2.xml")
    filt2= cv2.text.createERFilterNM2(filtC2)
    regions = cv2.text.detectRegions(input, filt1, filt2)
    return regions
   


img2 = cv2.imread("C:/Users/sergm/source/repos/OCRTry/Captura.png")
binarizada = Binarization(img2);
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
dilated = cv2.dilate(binarizada, kernel, iterations=5)
dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
imgCount, cnts, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
plt.gray()    
imgGrisPrueba= cv2.imread("C:/Users/sergm/source/repos/OCRTry/carta_entera.jpg") 
 

for i in range(len(areas)):
   box= cv2.boxPoints(areas[i])
   box= np.int0(box)
   cv2.drawContours(imgGrisPrueba,[box],0,(0,0,255),2)

   cropped = skewAndCrop(imgGrisPrueba, areas[i])
   text = identifyTextTesseract(cropped)
   print(text)
   plt.imshow(cropped)
   plt.show()
   cv2.waitKey(0)


text = identifyTextTesseract(imgGrisPrueba)
print(text)

plt.imshow(imgGrisPrueba)
plt.show()

