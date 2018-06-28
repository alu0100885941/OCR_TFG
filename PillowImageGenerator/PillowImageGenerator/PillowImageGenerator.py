from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import random as rand
import string as str
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc
import cv2
import os
import numpy as np
import imutils
iter=0
iter2=0
n_pruebas= 100

def noise_generator (noise_type,image):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    print(image.shape)
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image

def sp_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = rand.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def create_imgText(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/blanco_grande.jpg")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 40)
    n1= rand.randint(0,200)
    n2= rand.randint(n1,200)
    if(n2 < 100):
        n2=n2+100
    string=""

    for i in range(0,3):
        string+= rand.choice(str.ascii_letters)
    string = string.upper()
    if(n1<10):
        draw.text((2,2), ' 00%d/%d\n %s * EN' % (n1,n2, string), (0,0,0), font=font)
    elif(n1 < 100):
        draw.text((2,2), ' 0%d/%d\n %s * EN' % (n1,n2, string), (0,0,0), font=font)
    else:
        draw.text((2,2), ' %d/%d\n %s * EN' % (n1,n2, string), (0,0,0), font=font)

    
    img.save("C:/Users/sergm/source/repos/OCRTry/Generadas/Base/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/OCRTry/Generadas/Base/test%d.png" %(iter))
    
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-3,3)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/OCRTry/Generadas/Samples/test%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/OCRTry/Generadas/Answers/ans%d.txt" %(iter), "w")
   
    file.write('0%d/%d\n%s * EN' % (n1,n2, string))
    file.close
    
def create_dataTraining(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 15)
    n1= rand.randint(0,200)
    n2= rand.randint(n1,200)
    if(n2 < 100):
        n2=n2+100
    
    string=""

    for i in range(0,3):
        string+= rand.choice(str.ascii_letters)
    string = string.upper()
    if(n1<10):
        draw.text((2,2), '00%d/%d\n%s * EN' % (n1,n2, string), (0,0,0), font=font)
    elif(n1 < 100):
        draw.text((2,2), '0%d/%d\n%s * EN' % (n1,n2, string), (0,0,0), font=font)
    else:
        draw.text((2,2), '%d/%d\n%s * EN' % (n1,n2, string), (0,0,0), font=font)

    
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/train%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/train%d.png" %(iter))
   
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-5,5)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/train%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/ans%d.txt" %(iter), "w")
   
    file.write('0%d/%d\n%s * EN' % (n1,n2, string))
    file.close

def create_dataletrasTrain(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada60.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 25)
    string=""
    string= rand.choice(str.ascii_letters)
    string = string.upper()
    draw.text((20,20), '%s' % (string), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/test%d.png" %(iter))
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-5,5)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Modified/test%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/ans%d.txt" %(iter), "w")
    file.write('%s' % (string))
    file.close

def create_dataNumerosTrain(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada60.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 25)
    n1= rand.randint(0,9)
    draw.text((20,20), '%d' % (n1), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/test%d.png" %(iter))
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-5,5)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Modified/test%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/ans%d.txt" %(iter), "w")
    file.write('%d' % (n1))
    file.close

def create_dataletrasTest(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada60.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 25)
    string=""
    string= rand.choice(str.ascii_letters)
    string = string.upper()
    draw.text((20,20), '%s' % (string), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Unmodified/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Unmodified/test%d.png" %(iter))
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-5,5)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Modified/test%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers/ans%d.txt" %(iter), "w")
    file.write('%s' % (string))
    file.close

def create_dataNumerosTest(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada60.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 25)
    n1= rand.randint(0,9)
    draw.text((20,20), '%d' % (n1), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Unmodified/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Unmodified/test%d.png" %(iter))
    noised_img=sp_noise(img,0.01)
    rot_ang= rand.randint(-5,5)
    noised_img = imutils.rotate(noised_img,rot_ang)
    cv2.imwrite("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Modified/test%d.png" %(iter), noised_img)
    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers/ans%d.txt" %(iter), "w")
    file.write('%d' % (n1))
    file.close


#-------------------#
def create_dataNumerosTrain2(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 15)
    n1= rand.randint(0,1)
    draw.text((10,10), '%d' % (n1), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/NUM/test%d.png" %(iter))
    img= cv2.imread("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Samples/Unmodified/NUM/test%d.png" %(iter))

    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Train/Answers/NUM/ans%d.txt" %(iter), "w")
    file.write('%d' % (n1))
    file.close
def create_dataNumerosTest2(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco2_cuadrada.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Relay-Medium.ttf", 15)
    n1= rand.randint(0,1)
    draw.text((10,10), '%d' % (n1), (2,2,2), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Samples/Unmodified/NUM/test%d.png" %(iter))

    file = open("C:/Users/sergm/source/repos/PillowImageGenerator/Output/Test/Answers/NUM/ans%d.txt" %(iter), "w")
    file.write('%d' % (n1))
    file.close
#-------------------#
def arbiter_Train(iter, sub_iter):
    if(sub_iter%2==0):
        create_dataletrasTrain(iter)
    else:
        create_dataNumerosTrain(iter)

def arbiter_Test(iter, sub_iter):
    if(sub_iter%2==0):
        create_dataletrasTest(iter)
    else:
        create_dataNumerosTest(iter)

for i in range(0, n_pruebas):
 
    for j in range(0,80):
        #create_dataTraining(iter, j)
        arbiter_Train(iter,j)
        #create_dataNumerosTrain2(iter)
        
        iter=iter+1

    for j in range(0,20):
        #create_imgText(iter2, j)
        arbiter_Test(iter2, j)
        #create_dataNumerosTest2(iter2)
        iter2=iter2+1
    print("Prueba ", i)
    #create_imgText(i)
                
   



