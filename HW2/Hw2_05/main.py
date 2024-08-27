from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
from scipy import signal
from scipy import ndimage
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randrange
import glob
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.showImage)
        self.pushButton_3.clicked.connect(self.classDistribution)
        self.pushButton_4.clicked.connect(self.modleStructure)
        self.pushButton_5.clicked.connect(self.accuracyComparision)
        self.pushButton_6.clicked.connect(self.inference)

        self.imageCat = []
        self.imageDog = []
        
    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.path = fname[0]
        #img= Image.open(self.path) 
        #self.img = img.resize((224, 224), Image.ANTIALIAS)
        self.img = cv2.imread(fname[0], -1)
        self.img = cv2.resize(self.img, (224,224))
        if self.img.size == 1:
            return 

        height, width, channel = self.img.shape #self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))



    def classDistribution(self):
        #img = mpimg.imread("count.png")
        #imgplot = plt.imshow(img)
        #plt.show()
        image = Image.open("count.png")
        image.show()

    def showImage(self):
        if len(self.imageCat) == 0:
            for filename in glob.glob(os.path.join("inference_dataset\Cat", "*.jpg")):
                image = Image.open(filename)
                self.imageCat.append(image)
        
        if len(self.imageDog) == 0:
            for filename in glob.glob(os.path.join("inference_dataset\Dog", "*.jpg")):
                image = Image.open(filename)
                self.imageDog.append(image)

        image1 = self.imageCat[ randrange(len(self.imageCat)) ].copy()
        image2 = self.imageDog[ randrange(len(self.imageDog)) ].copy()
        
        image1 = image1.resize((224, 224))
        image2 = image2.resize((224, 224))  

 
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image1)
        plt.title("Cat")
        plt.subplot(122)
        plt.imshow(image2)
        plt.title("Dog")
        plt.show()

    def modleStructure(self):
        IMAGE_SIZE = (224, 224)
        net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

        x = net.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        net_final = Model(inputs=net.input, outputs=output_layer)

        print(net_final.summary())

    def accuracyComparision(self):
        image = Image.open("accuracy.png")
        image.show()

    def inference(self):
        net = load_model('model-resnet50-loss1.h5')
        class_list = ['Cat', 'Dog']

        img= Image.open(self.path) 
        img = img.resize((224, 224), Image.ANTIALIAS)


        x = image.img_to_array(img)

        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        print(pred)
        if pred < 0.44:
            print("predict: cat")
            title = "class: cat"
        else:
            print("predict: dog")
            title = "class: dog"


        plt.title(title)
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())