from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import random as rand
import string as str

iter=0

def create_imgText(iter):
    img = Image.open("C:/Users/sergm/source/repos/PillowImageGenerator/imagen_blanco.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Users/sergm/source/repos/PillowImageGenerator/Aaargh.ttf", 32)
    n1= rand.randint(0,200)
    n2= rand.randint(n1,200)
    string=""

    for i in range(0,3):
        string+= rand.choice(str.ascii_letters)
    draw.text((10,10), '%d/%d\n%s * EN' % (n1,n2, string), (0,0,0), font=font)
    img.save("C:/Users/sergm/source/repos/PillowImageGenerator/Output/prueba_salida%d.png" %(iter))

for i in range(0,10000):
    create_imgText(iter)
    iter+=1



