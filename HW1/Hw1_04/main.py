from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
import keyboard
from matplotlib import pyplot as plt


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.loadImage1)        
        self.pushButton_2.clicked.connect(self.loadImage2)
        self.pushButton_3.clicked.connect(self.findKeypoint)  
        self.pushButton_4.clicked.connect(self.MatchKeypoint)

    
    def loadImage1(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")     
        self.img1 = cv2.imread(fname[0])
        self.path1 = fname[0]

    def loadImage2(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")     
        self.img2 = cv2.imread(fname[0])
        self.path2 = fname[0]

    def findKeypoint(self):
        #reading image
        img1 = self.img1
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        #keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

        img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1,color=(0,255,0))
        cv2.imshow('find Keypoint',img_1)
        

    def MatchKeypoint(self):
        # read images
        img1 = self.img1 
        img2 = self.img2

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1_1 = cv2.imread(self.path1)  # 原圖1
        img2_1 = cv2.imread(self.path2)  # 原圖2
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_1, None)
        kp2, des2 = sift.detectAndCompute(img2_1, None)
        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)#4~~~

        good = [[m] for m, n in matches if m.distance < 0.7*n.distance]
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                                  matchColor=(0, 255, 255), matchesMask=None,
                                  singlePointColor=(0, 255, 0), flags=0)
        cv2.imshow('match_keypoints',img3)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())