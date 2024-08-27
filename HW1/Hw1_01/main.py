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

        self.pushButton_1.clicked.connect(self.open_folder)        
        self.pushButton_2.clicked.connect(self.loadImage_L)
        self.pushButton_3.clicked.connect(self.loadImage_R)        
        
        self.pushButton.clicked.connect(self.corner)
        self.pushButton_4.clicked.connect(self.intrinsic)
        self.pushButton_5.clicked.connect(self.extrinsic)
        self.pushButton_6.clicked.connect(self.distortion)
        self.pushButton_7.clicked.connect(self.result)
        
        self.pushButton_8.clicked.connect(self.wordsOnBoard)        
        self.pushButton_9.clicked.connect(self.wordsVertically)
        
        self.pushButton_10.clicked.connect(self.stereo)
        
        self.center = None
        self.path = './Dataset_CvDl_Hw1/Q1_Image'
        self.char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
            [7,5,0], # slot 1
            [4,5,0], # slot 2
            [1,5,0], # slot 3
            [7,2,0], # slot 4
            [4,2,0], # slot 5
            [1,2,0]  # slot 6
        ]

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,"Open folder")
        print(folder_path)
        self.path = folder_path
        

    def loadImage_L(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.imgL = cv2.imread(fname[0])
        #cv2.imshow("Load Image", self.imgL)

    def loadImage_R(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.imgR = cv2.imread(fname[0])
        #cv2.imshow("Load Image", self.imgR)

    def calibration(self, images:list, width_board = 11, height_board = 8):
    
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        objp = np.zeros((height_board*width_board, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width_board, 0:height_board].T.reshape(-1, 2)

        # Array to store object points and image points from all the image.
        objpoints = [] #3d point in real world space
        imgpoints = [] #2d points in image plane

        for index, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corner = cv2.findChessboardCorners(gray, (width_board, height_board), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corner)
        return objpoints, imgpoints

    #problem 1
    def corner(self):
        cv2.namedWindow("Chessboard", cv2.WINDOW_NORMAL)
        imageList = []

        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane

        
        for i in range(1,16):
            # Load the image
            #image = cv2.imread("./Dataset_CvDl_Hw1/Q1_Image/" + str(i) + ".bmp")
            image = cv2.imread(self.path + '/' + str(i) + ".bmp")
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            imageList.append(image)   
            

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            retval, corners = cv2.findChessboardCorners(gray, (8,11), flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

            
            if retval:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                chessboardImage = cv2.drawChessboardCorners(image, (8,11), corners2, retval)            
            else:
                print("No Checkerboard Found")

        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)      
        self.ret = ret  
        self.instrincMatrix = cameraMatrix        
        self.distortion = dist
        self.rotationalvector = rvecs
        self.translationalvectors = tvecs    

        i = 0
        while True:
            i = (i + 1) % 15
            cv2.imshow("Chessboard", imageList[i])
            cv2.waitKey(500)

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                break
            if keyboard.is_pressed("p"):
                cv2.waitKey(0)

    def intrinsic(self):
        print(self.instrincMatrix)
    
    def extrinsic(self):
        index = self.spinBox.value()
        #print(index)
        index = index - 1

        R = np.zeros((3, 3))
        cv2.Rodrigues(self.rotationalvector[index],R)

        extrinsic = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                if j != 3:
                    extrinsic[i,j] = R[i,j]
                else:
                    extrinsic[i,j] = self.translationalvectors[index][i]
        print(extrinsic)

    def distortion(self):
        print(self.distortion)

    def result(self):
        ret = self.ret
        mtx = self.instrincMatrix      
        dist = self.distortion
        rvecs = self.rotationalvector
        tvecs = self.translationalvectors
        undistoredList = []

        for i in range(15):
            #take a new image
            img = cv2.imread(self.path + '/' + str(i+1) + ".bmp")
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            
            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            #cv2.cv2.imshow('calibresult',dst)
            undistoredList.append(dst)

        distoredList = []
        for i in range(1,16):
            image = cv2.imread(self.path + '/' + str(i) + ".bmp")
            image = cv2.resize(image, (493, 493), interpolation=cv2.INTER_AREA)
            distoredList.append(image)   

        i = 0
        while True:
            i = (i + 1) % 15
            numpy_horizontal = np.hstack((distoredList[i], undistoredList[i]))
            cv2.imshow("Distored vs Undistored", numpy_horizontal)
            cv2.waitKey(500)

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                break
            if keyboard.is_pressed("p"):
                cv2.waitKey(0)

    #problem 2
    def draw_char(self, img, char_list:list):
        draw_image = img.copy()
        for line in char_list:
            line = line.reshape(2,2)
            draw_image = cv2.line(draw_image, tuple(line[0]), tuple(line[1]), (0,255,0), 10, cv2.LINE_AA)
        return draw_image   

    def wordsOnBoard(self):
        string = self.lineEdit.text()
        string = string[:6]

        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        

        if not string.isupper():
            string = string.upper()
        
        chCordinateList = []

        # load image
        imageList = []
        for i in range(1,6):
            image = cv2.imread(self.path + '/' + str(i) + ".bmp")
            imageList.append(image)
            
        self.q2_objps, self.q2_imageps = self.calibration(imageList)
        for index, image in enumerate(imageList):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)

            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = self.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv2.namedWindow("WORD ON BOARD", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("WORD ON BOARD", draw_image)
                cv2.waitKey(1000)
        

    def wordsVertically(self):       
        string = self.lineEdit.text()
        string = string[:6]

        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        

        if not string.isupper():
            string = string.upper()
        
        chCordinateList = []

        # load image
        imageList = []
        for i in range(1,6):
            image = cv2.imread(self.path + '/' + str(i) + ".bmp")
            imageList.append(image)
            
        self.q2_objps, self.q2_imageps = self.calibration(imageList)
        for index, image in enumerate(imageList):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)

            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = self.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv2.namedWindow("WORD VERTICAL", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("WORD VERTICAL", draw_image)
                cv2.waitKey(1000)

    def stereo(self):
        imgL = self.imgL #cv2.imread("./Dataset_CvDl_Hw1/Q3_Image/imL.png")
        imgR = self.imgR #cv2.imread("./Dataset_CvDl_Hw1/Q3_Image/imR.png")
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        

        # Creating an object of StereoBM algorithm
        stereo = cv2.StereoBM_create(numDisparities = 256, blockSize = 25)      

        disparity = stereo.compute(grayL,grayR)

        disparity = cv2.normalize(disparity, None, alpha=0,  beta=255, norm_type=cv2.NORM_MINMAX,  dtype=cv2.CV_8UC1)
        cv2.namedWindow("Disparity", cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("Disparity", int(disparity.shape[1]/4), int(disparity.shape[0]/4))
        cv2.imshow("Disparity", disparity)
        cv2.waitKey(1000)

        print(disparity.shape)# disparity.shape = (1848, 2724)



        cv2.namedWindow("Checking Disparity", cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("Checking Disparity", int(disparity.shape[1]/4*2), int(disparity.shape[0]/4))
        self.map_disparity(imgL, imgR, disparity, "Checking Disparity")
        cv2.waitKey(0)

    def map_disparity(self, imgL, imgR, disparity, win):
        hL, wL = imgL.shape[:2]        
        merge_image = self.concat_image(imgL, imgR)
        cur_img = merge_image.copy()

        def onmouse(event, x, y, flags, param):
            
            nonlocal cur_img
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cur_img = merge_image.copy()
                cur_x, cur_y = x, y
                if cur_x < 0:
                    cur_x = 0
                elif cur_x >= wL:
                    cur_x = wL-1
                if cur_y < 0:
                    cur_y = 0
                elif cur_y >= hL:
                    cur_y = hL-1
                delta_pos = disparity[cur_y,cur_x]
                print("disparity value at (x,y) = ({},{}): {}".format(cur_x, cur_y, delta_pos))
                
                if delta_pos != 0:
                    x_right = cur_x -delta_pos + wL
                    cur_img = cv2.circle(cur_img, (x_right,cur_y), radius=20, color=(0, 255, 0), thickness=-1)
                
            cv2.imshow(win, cur_img)
        cv2.setMouseCallback(win, onmouse) 

    def concat_image(self, src, dst):
        merge = cv2.hconcat([src, dst])
        return merge
        
        
        
            


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())