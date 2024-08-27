from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
import os
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.loadVideo)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton_3.clicked.connect(self.open_folder)
        self.pushButton_4.clicked.connect(self.backgroundSubtraction)
        self.pushButton_5.clicked.connect(self.preprocessing)
        self.pushButton_6.clicked.connect(self.tracking)
        self.pushButton_7.clicked.connect(self.perspectiveTransform)
        self.pushButton_8.clicked.connect(self.imageReconstruction)
        self.pushButton_9.clicked.connect(self.reconstructionError)



        self.points = None


    def loadVideo(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")

        cap= cv2.VideoCapture(fname[0])
        self.video = cap
   
    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.img = cv2.imread(fname[0])


    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,"Open folder")
        self.path = folder_path

    #problem1
    def backgroundSubtraction(self):
        backSub = cv2.createBackgroundSubtractorMOG2()
        

        cap = self.video
        

        build_model = False

        frameList =[]
        while cap.isOpened():
            ret, frame = cap.read()

            
            
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)


            if len(frameList) < 25:
                #collect 0~24 frames
                frameList.append(gray)
            else:
                if not build_model:
                    # build a gaussian model 
                    frames = np.array(frameList)
                    mean = np.mean(frames, axis= 0)
                    standard = np.std(frames, axis=0)
                    standard[standard < 5] = 5
                    build_model = True
                else:
                    mask[np.abs(gray - mean) > standard*5] = 255

            
            #gray2RGB
            mask_out = np.stack((mask,)*3, axis=-1)
            #mask
            result = cv2.bitwise_and(frame, frame,mask=mask)

            output = np.concatenate([frame, mask_out, result], axis=1)
            cv2.imshow('problem1', output)
            if cv2.waitKey(40) == ord('q'):
                break        

        cap.release()
        cv2.destroyAllWindows()

    #problem2
    def preprocessing(self):
        cap = self.video

        ret, frame = cap.read()

        cv2.imshow('problem2_original', frame)
        self.keypoints_p0 = []




        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        params = cv2.SimpleBlobDetector_Params()
        # 设置阈值
        params.minThreshold = 10
        params.maxThreshold = 200
        # 设置选择区域
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        # 设置圆度
        params.filterByCircularity = True
        params.minCircularity = 0.75
        # 设置凸度
        params.filterByConvexity = True
        params.minConvexity = 0.7
        # 设置惯性比
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)
 
        
        # Detect blobs.
        keypoints = detector.detect(im)


        p0 = []
        for item in keypoints :
            #(x,y) = (item.pt[0], item.pt[1])
            p0.append([item.pt[0], item.pt[1]])
        p0 = np.array(p0, np.int)
        self.points = p0
        ans = self.drawBoundingBox(p0, frame.copy())
        


    def drawBoundingBox(self, points, img):
        #img = np.array(img, np.uint8)
        for a in points:
            x, y  = a
            cv2.line(img, (x-6, y), (x+6, y), (0, 0, 255), 1)
            cv2.line(img, (x, y-6), (x, y+6), (0, 0, 255), 1)
            cv2.rectangle(img, (x-6, y-6), (x+6, y+6), (0, 0, 255), 1)


            cv2.imshow("problem2.1", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def tracking(self):
        cap = self.video
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        p0 = np.expand_dims(np.array(self.points),axis=1)
        p0 = p0.astype(np.float32)

        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while(1):
            ret,frame = cap.read()
            
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                #降維至一維陣列
                a,b = new.ravel()
                c,d = old.ravel()

                mask = cv2.line(mask, (a,b),(c,d),np.array([0,0,255]).tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,np.array([0,0,255]).tolist(),-1)
            img = cv2.add(frame,mask)
            cv2.imshow('problem2.2',img)

            if (cv2.waitKey(30) & 0xff) == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

    #problem 3
    def perspectiveTransform(self):
        cap = self.video
        image = self.img
        h, w = image.shape[:2]


        #Loading one of the predefined distionaries in aruco module
        #This DICT_4X4_250 dictionary is composed of 250 markers and marker size of 4X4 bits
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        # Initial parameters for the detectmaker process
        param = cv2.aruco.DetectorParameters_create()

        cv2.namedWindow("problem3", cv2.WINDOW_GUI_EXPANDED)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Detect Aruco makers in image and get the content of each marker
                markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                    frame, 
                    dictionary,
                    parameters = param
                )
                

                #Find id for each markers
                id1 = np.squeeze(np.where(markerIds == 1))
                id2 = np.squeeze(np.where(markerIds == 2))
                id3 = np.squeeze(np.where(markerIds == 3))
                id4 = np.squeeze(np.where(markerIds == 4))
                final = np.zeros_like(frame)
                
                # Process of perspective transform
                #if id1 != [] and id2 != [] and id3 != [] and id4 != []:
                if id1.size > 0 and id2.size > 0 and id3.size > 0 and id4.size > 0:
                    # Check if all markers can be detect or not
                    # Get the top-left corner of marker1 
                    pt1 = np.squeeze(markerCorners[id1[0]])[0]
                    # Get the top-right corner of marker2
                    pt2 = np.squeeze(markerCorners[id2[0]])[1]
                    # Get the bottom-right corner of marker3
                    pt3 = np.squeeze(markerCorners[id3[0]])[2]
                    # Get the bottom-left corner of marker4
                    pt4 = np.squeeze(markerCorners[id4[0]])[3]
                    

                    # Get coordinates of the corresponding quadrangle vertices in the destination image
                    pts_dst = [[pt1[0], pt1[1]]]
                    pts_dst += [[pt2[0], pt2[1]]]
                    pts_dst += [[pt3[0], pt3[1]]]
                    pts_dst += [[pt4[0], pt4[1]]]

                    #Get coordinates of quadrangle  vertices in the source image
                    pts_src = [[0, 0], [w, 0], [w, h], [0, h]]

                    retval, mask = cv2.findHomography(np.asfarray(pts_src), np.asfarray(pts_dst))
                    out = cv2.warpPerspective(image, retval, (frame.shape[1], frame.shape[0]))


                    mask_image = np.zeros_like(out)
                    mask_image[(out[:,:,:]*255) > 0] = 255

                    mask_image = cv2.bitwise_not(mask_image)
                    frame_mask = cv2.bitwise_and(frame, mask_image)
                    final = cv2.bitwise_or(frame_mask, out)
                    output = np.concatenate([frame, final], axis=1)
                    cv2.imshow("problem3", output)

                key = cv2.waitKey(25) & 0xFF
                if key == ord("q"):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    #problem 4
    def  imageReconstruction(self):      
        
        imageList = []
        inputList = []
        for filename in glob.glob(os.path.join(self.path, "*.jpg")):
            image = cv2.imread(filename)            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imageList.append(image)
            inputList.append(image.flatten())

        self.images = imageList

        h, w, c = imageList[0].shape

        #PCA & inverse PCA
        pca = PCA(n_components = int(len(imageList) * 0.8))
        components = pca.fit_transform(inputList)
        reconstructionImage = pca.inverse_transform(components)

        #normalize
        normalizedImg = np.zeros((len(imageList), h*w*c))
        normalizedImg = cv2.normalize(reconstructionImage,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        self.reconstruction = normalizedImg

        fig, ax = plt.subplots(4, 15, 
                                figsize= (9, 5),
                                subplot_kw={'xticks': [], 'yticks'    : []},
                                gridspec_kw = dict(hspace=0.1, wspace=0.1))
        for i in range(0, 15):
            ax[0, i].imshow(self.images[i].reshape(h, w, 3))
            ax[1, i].imshow(np.reshape(self.reconstruction[i, :].astype(np.uint8), (h, w, 3)))
            ax[2, i].imshow(self.images[i + 15].reshape(h, w, 3))
            ax[3, i].imshow(np.reshape(self.reconstruction[i + 15, :].astype(np.uint8), (h, w, 3)))
        ax[0, 0].set_ylabel('Original')
        ax[1, 0].set_ylabel('Reconstruction')
        ax[2, 0].set_ylabel('Original')
        ax[3, 0].set_ylabel('Reconstruction')
        plt.show()





    def reconstructionError(self):
        h,w = self.images[0].shape[0:2]
        computing = []
        for origin_image, reconstruction in zip(self.images, self.reconstruction):
            orig_img = origin_image.reshape(h,w,3)
            orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

            recons_img = np.reshape(reconstruction.astype(np.uint8), (h,w,3))
            recons_gray = cv2.cvtColor(recons_img, cv2.COLOR_RGB2GRAY)
            
            arr_sub = np.subtract(orig_gray, recons_gray)
            arr_square = np.multiply(arr_sub, arr_sub)
            arr_sum = np.sum(arr_square)
            sum_error = np.sqrt(arr_sum)
            computing.append(round(sum_error))
        print("reconstruction error :")
        print("max error: " + str(np.max(computing)))
        print("min error: " + str(np.min(computing)))








        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())