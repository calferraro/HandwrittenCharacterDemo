# Initial setup
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from tensorflow.keras.utils import to_categorical  # Will be used to categorize an output layer neurons

# Import all required layers types
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
from keras.datasets import mnist 

# Load optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

# Imports for gui
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import os

from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *

import argparse

case = [
    ['conv1', 'conv2', 'pool', 'dropout1', 'flatten', 'dense1', 'dropout2', 'dense2'],
    ['conv1', 'pool', 'conv2', 'pool', 'dropout1', 'flatten', 'dense1', 'dropout2', 'dense2'],
    ['conv1', 'conv2', 'pool', 'flatten', 'dense1', 'dense2'],
    ['conv1', 'pool', 'conv2', 'pool', 'flatten', 'dense1', 'dense2'],
    ['conv1', 'conv2', 'pool', 'flatten', 'dense1', 'dropout2', 'dense2'],
    ['conv1', 'pool', 'conv2', 'pool', 'flatten', 'dense1', 'dropout2', 'dense2']
]

def CNN_modelSetup(args, layers):
  CNN_model = Sequential()

  for l in layers:
    if(l == 'conv1'):
      CNN_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))) #conv1
    if(l == 'conv2'):
      CNN_model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')) #conv2
    if(l == 'pool'):
      CNN_model.add(MaxPooling2D((2, 2))) #pool1
    if(l == 'dropout1'):
      CNN_model.add(Dropout(0.25))
    if(l == 'flatten'):
      CNN_model.add(Flatten())
    if(l == 'dense1'):
      CNN_model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    if(l == 'dropout2'):
      CNN_model.add(Dropout(0.5))
    if(l == 'dense2'):
      CNN_model.add(Dense(10 if args.isDigit else 26, activation='softmax'))
	# compile model
  opt = Adam(learning_rate=0.001)
  # opt = SGD(learning_rate=0.01, momentum=0.9)
  CNN_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return CNN_model

def character_gui(case):
    classes_letter=['a','b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    width = 300
    height = 300
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)

    modelCNN=tf.keras.models.load_model('./models/LetterRecognition_case{}.h5'.format(case)) # Load our model

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
        draw.line([x1, y1, x2, y2],fill="black",width=10)
        
    def model():
        filename = "image.png"
        image1.save(filename)
        pred=testing()
        
        txt.insert(tk.INSERT,"Prediction: {}".format(classes_letter[pred[0]]))
        
    def clear():
        cv.delete('all')
        draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
        txt.delete('1.0', END)
        
    def testing():
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)
        #cv2.imshow('img',img)
        img=cv2.resize(img,(28,28))
        img=img.reshape(1,28,28,1)
        img=img.astype('float32')
        img=img/255.0
        
        pred=modelCNN.predict(img)
        classes_x=np.argmax(pred,axis=1)
        
        
        return classes_x
        
        
    root = Tk()
    ##root.geometry('1000x500') 

    root.resizable(0,0)
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    txt=tk.Text(root,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
                padx=10,pady=10,height=5,width=20)

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    ##button=Button(text="save",command=save)
    btnModel=Button(text="Predict",command=model)
    btnClear=Button(text="clear",command=clear)
    ##button.pack()
    btnModel.pack()
    btnClear.pack()
    txt.pack()
    root.title('Letter recognition')
    root.mainloop()

def digit_gui(case):
  classes=[0,1,2,3,4,5,6,7,8,9]
  width = 300
  height = 300
  center = height//2
  white = (255, 255, 255)
  green = (0,128,0)

  modelCNN=tf.keras.models.load_model('./models/DigitRecognition_case{}.h5'.format(case)) # Load our model

  def paint(event):
      x1, y1 = (event.x - 10), (event.y - 10)
      x2, y2 = (event.x + 10), (event.y + 10)
      cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
      draw.line([x1, y1, x2, y2],fill="black",width=10)
      
  def model():
      filename = "image.png"
      image1.save(filename)
      pred=testing()
      
      txt.insert(tk.INSERT,"Prediction: {}".format(str(pred[0]))) 
      
  def clear():
      cv.delete('all')
      draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
      txt.delete('1.0', END)
      
  def testing():
      img=cv2.imread('image.png',0)
      img=cv2.bitwise_not(img)
      #cv2.imshow('img',img)
      img=cv2.resize(img,(28,28))
      img=img.reshape(1,28,28,1)
      img=img.astype('float32')
      img=img/255.0
      
      pred=modelCNN.predict(img)
      classes_x=np.argmax(pred,axis=1)
      
      
      return classes_x
      
      
  root = Tk()
  ##root.geometry('1000x500') 

  root.resizable(0,0)
  cv = Canvas(root, width=width, height=height, bg='white')
  cv.pack()

  # PIL create an empty image and draw object to draw on
  # memory only, not visible
  image1 = PIL.Image.new("RGB", (width, height), white)
  draw = ImageDraw.Draw(image1)

  txt=tk.Text(root,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
              padx=10,pady=10,height=5,width=20)

  cv.pack(expand=YES, fill=BOTH)
  cv.bind("<B1-Motion>", paint)

  ##button=Button(text="save",command=save)
  btnModel=Button(text="Predict",command=model)
  btnClear=Button(text="clear",command=clear)
  ##button.pack()
  btnModel.pack()
  btnClear.pack()
  txt.pack()
  root.title('Digit recognition')
  root.mainloop()


def main(args):
  if(args.isDigit):
    # do digit recognition code here
    # Data load and preprocess
    (Xtr_0, Ytr_0), (Xts_0, Yts_0) = mnist.load_data()

    Xtr = Xtr_0.reshape((Xtr_0.shape[0], 28, 28, 1))  # Training input reshape matrix -> vector
    Xts = Xts_0.reshape((Xts_0.shape[0], 28, 28, 1))  # Testing input reshape matrix -> vector
      
    Ytr = to_categorical(Ytr_0) # Represent an output as a vecotr 1x10; exp: 4 -> [0 0 0 0 1 0 0 0 0 0]
    Yts = to_categorical(Yts_0) # Same for test data

    # Data normalization
    Xtr_norm = Xtr.astype('float32')/255.0 
    Xts_norm = Xts.astype('float32')/255.0 
    
    if(args.train):
      # call training function here
      # Model training
      CNN_mod = CNN_modelSetup(args, case[args.case])
      CNN_mod.fit(Xtr_norm, Ytr, validation_data=(Xts_norm, Yts), epochs=15, batch_size=100, verbose=2)
      scores = CNN_mod.evaluate(Xts_norm, Yts, verbose=0)
    
      CNN_mod.save('./models/DigitRecognition_case{}.h5'.format(args.case))
    else:
      # call gui function here
      digit_gui(args.case)
  else:
     if(args.train):

            # do character recognition code here
            with open("./data/latin_data.csv") as file_name:
                X_all_0 = np.loadtxt(file_name, delimiter=",")
                
            X_all=X_all_0.reshape(X_all_0.shape[0], 28, 28, 1)
            X_all=X_all.astype('float32')            
            
            with open("./data/latin_label.csv") as file_name:
                Y_all_0 = np.loadtxt(file_name, delimiter=",")
         
            Y_all=to_categorical(Y_all_0)

            numbyclass=[493, 496, 508, 461, 500, 482, 509, 509, 476, 458, 460, 523, 547, 521, 465, 526, 484, 470, 519, 477, 475, 480, 465, 490, 545, 483]
            Trnumbyclass=[394, 397, 406, 369, 400, 386, 407, 407, 381, 366, 368, 418, 438, 417, 372, 421, 387, 376, 415, 382, 380, 384, 372, 392, 436, 386]

            currentidx=0
            tridx=[]
            tsidx=[]
            for i in range(26):
                tridx=np.append(tridx,range(currentidx, currentidx+Trnumbyclass[i]))
                tsidx=np.append(tsidx,range(currentidx+Trnumbyclass[i], currentidx+numbyclass[i]))
                currentidx=currentidx+numbyclass[i]

            tridx=tridx.astype(int)   

            tsidx=tsidx.astype(int) 


            Xtr=np.array([X_all[x] for x in tridx])
            Xts=np.array([X_all[x] for x in tsidx])
            Xtr_0=np.array([X_all_0[x] for x in tridx])
            Xts_0=np.array([X_all_0[x] for x in tsidx])
            Ytr=np.array([Y_all[x] for x in tridx])
            Yts=np.array([Y_all[x] for x in tsidx])
            Ytr_0=np.array([Y_all_0[x] for x in tridx])
            Yts_0=np.array([Y_all_0[x] for x in tsidx])

     
            print('start training')
          # call training function here
            CNN_mod = CNN_modelSetup(args, case[args.case])
            CNN_mod.fit(Xtr, Ytr, validation_data=(Xts, Yts), epochs=15, batch_size=50, verbose=2)
            scores = CNN_mod.evaluate(Xts, Yts, verbose=0)

            CNN_mod.save('./models/LetterRecognition_case{}.h5'.format(args.case))
     else:
            numbyclass=[493, 496, 508, 461, 500, 482, 509, 509, 476, 458, 460, 523, 547, 521, 465, 526, 484, 470, 519, 477, 475, 480, 465, 490, 545, 483]
            Trnumbyclass=[394, 397, 406, 369, 400, 386, 407, 407, 381, 366, 368, 418, 438, 417, 372, 421, 387, 376, 415, 382, 380, 384, 372, 392, 436, 386]
          # call gui function here
            character_gui(args.case)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--train', action="store_true")
  parser.add_argument('--case', type=int, default=0)
  parser.add_argument('--isDigit', action="store_true")

  args = parser.parse_args()

  main(args)
  