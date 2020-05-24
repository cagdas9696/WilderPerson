import sys
import os
import pandas as pd 



train_y = os.listdir("Annotations/") 
images = os.listdir("Images")
img=[]

 

'''
while i<13382:
    #print("cagdas")
    j=0
    while j<9000:
        if images[i][0] != train_y[j][0]:
            j=j+1
            continue        
        elif  images[i][1] != train_y[j][1]:
            j=j+1
            continue        
        elif  images[i][2] != train_y[j][2]:
            j=j+1
            continue        
        elif  images[i][3] != train_y[j][3]:
            j=j+1
            continue        
        elif  images[i][4] != train_y[j][4]:
            j=j+1
            continue        
        elif  images[i][5] != train_y[j][5]:
            j=j+1
            continue        
        else:
            img.append(images[i])
            j=j+1
            break
        j=j+1
    i=i+1  '''

import shutil

with open('test.txt', 'r') as f:
    names = [name.replace('\n','') for name in f.readlines()]


for name in names:
    image = 'images/' + name +'.jpg'
    shutil.copy(image,'test/')
    
    
    
    
    
    
    
    
    
    
    