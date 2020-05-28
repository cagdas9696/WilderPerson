# -*- coding: utf-8 -*-
"""
Created on Mon May 25 03:04:44 2020

@author: cagda
"""
import sys
import os
import pandas as pd 


'''
xx=''
with open('train/000042.jpg.txt', 'r') as f:
    liste=f.readlines()
    liste.pop(0)
    for i in liste:
        i=i[:-1]
        x=i.split(' ')
        xx+=x[1]+','+ x[2] + ',' + x[3] + ',' + x[4] + ',' + x[0] + ' '

'''



files=os.listdir('train')


for imagename in files:
    box_format=''
    if imagename.endswith('jpg'):
        txt_path = imagename.replace(".jpg",".jpg.txt")
        with open('train/'+txt_path,'r') as file:
            label=file.readlines()
            label.pop(0)
            for i in label:
                i=i[:-1]
                x=i.split(' ')
                box_format+=' ' + x[1]+','+ x[2] + ',' + x[3] + ',' + x[4] + ',' + str(int(x[0]) - 1) 
                
        with open('annotation.txt', 'a') as newline:
            newline.write('train/' + imagename + box_format + '\n')
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                