import sys
from collections import defaultdict
import numpy as np

import scipy.misc

from keras.utils import np_utils

from sklearn.preprocessing import label_binarize
from keras.preprocessing import image as image_utils

from keras.utils.np_utils import probas_to_classes
import timeit
from Tkinter import *
from tkFileDialog import *
import argparse
import cv2
import net

np.random.seed(1337)


fenetre = Tk()
fenetre.geometry('700x700')
fenetre.title('Breast Cancer Screening')

lbl = Label(fenetre, text="Breast Cancer Screening", font='size, 30')
lbl.pack()

img = PhotoImage(file='blank.png')
panel = Label(fenetre, image = img)
panel.pack()

def callback():
    global filepath, label
    filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
    img2 = PhotoImage(file=filepath)
    panel.configure(image = img2)
    panel.image = img2
    result= '?'
    label.config(text=result)

result= '?'

def Pred():
	global result,label
	image = cv2.imread(filepath)

	print("[INFO] loading and preprocessing image...")
	image = image_utils.load_img(filepath, target_size=(224, 224))
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)

	start = timeit.default_timer()
	# load the network
	print("[INFO] loading network...")
	model, tags_from_model = net.load("model")
	net.compile(model)

	# classify the image
	print("[INFO] classifying image...")
	preds = model.predict(image)	
	y_classes = probas_to_classes(preds)
	if (y_classes==1):
		print "it is cancerous"
		result="malignant"
	else:
		print "it is benign"
		result="benign"
	label.config(text=result)
	stop = timeit.default_timer()
	exec_time= stop - start
	print "Predicting Execution Time is %f s" % exec_time 
	return 

bouton=Button(fenetre, text="Cancel", command=fenetre.quit)
bouton.pack()

browsebutton = Button(fenetre, text="Browse", command=callback)
browsebutton.pack()

boutonP = Button(fenetre, text="Predict", command = Pred)
boutonP.pack()

page1 = Frame(fenetre, bg="ivory", width=200, height=200)
label = Label(page1, text=result, bg="ivory", font='size, 20')
label.pack(padx=10, pady=10)
page1.pack()

fenetre.mainloop()

