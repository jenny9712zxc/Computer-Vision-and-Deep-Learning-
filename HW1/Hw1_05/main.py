from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
import keyboard
from matplotlib import pyplot as plt


import torch                                          
import torchvision.models as models                   
from PIL import Image                                 
import torchvision.transforms.functional as TF        
from torchsummary import summary                                                      
from torchviz import make_dot 
import torchvision.transforms as transforms
import torchvision

from nn_module_vgg19 import VGG19

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import matplotlib.image as mpimg

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.loadImage)        
        self.pushButton_2.clicked.connect(self.trainingImage)
        self.pushButton_3.clicked.connect(self.modelStructure)  
        self.pushButton_4.clicked.connect(self.dataAugmentation)
        self.pushButton_5.clicked.connect(self.accuracyLoss)
        self.pushButton_6.clicked.connect(self.inference) 
    
    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.path = fname[0]
        #self.img = cv2.imread(fname[0])
        #self.img = cv2.resize(self.img, (32, 32), interpolation=cv2.INTER_AREA)
        
        #showimage
        image = cv2.imread(fname[0])
        height, width, channel = image.shape
        bytesPerline = 3 * width
        self.qImg = QImage(image.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))


        

    def trainingImage(self):    
        # load_cifar_10_data
        data_dir = "data/cifar-10-batches-py"
    
        meta_data_dict = unpickle(data_dir+"/batches.meta")
        #meta_key    [b'num_cases_per_batch', b'label_names', b'num_vis']
        cifar_label_names = meta_data_dict[b'label_names']    
        cifar_label_names = np.array(cifar_label_names)
        #[b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

        # training data
        cifar_train_data = None
        cifar_train_filenames = []
        cifar_train_labels = []


        datadict = unpickle(data_dir+"/test_batch")
        #print(datadict.keys())#batch_key  [b'batch_label', b'labels', b'data', b'filenames']

        

        X = datadict[b'data'] 
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)   
        cifar_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        plt.ion()
        #Visualizing CIFAR 10
        fig, axes1 = plt.subplots(3,3,figsize=(5,5))
        for j in range(3):
            for k in range(3):
                i = np.random.choice(range(len(X)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0]) 
                index = Y[i]
                index =  int(index)
                axes1[j][k].set_title(cifar_label_names[index])
    
    def modelStructure(self):     
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

              
        net = VGG19().to(device)    
        summary(net, (3, 32, 32))                


    def dataAugmentation(self):
        img_pil = Image.open(self.path) 
        #img_pil.show() 

 
        
        padding = (40, 40, 40, 40)
        transform = transforms.Compose([
            transforms.Resize((100,150)),
            transforms.Pad(padding, padding_mode="symmetric"), 
        ])
        new_img1 = transform(img_pil)
        plt.subplot(1,3,1)
        plt.imshow(new_img1)
        #new_img1.show() 

     

        transform = transforms.Compose([
            transforms.Resize((100,150)),
            transforms.RandomHorizontalFlip(p=0.9),
        ])

        new_img2 = transform(img_pil)
        plt.subplot(1,3,2)
        plt.imshow(new_img2)
        #new_img2.show()


        transform = transforms.Compose([
            transforms.Resize((100,150)),
            transforms.Grayscale(num_output_channels=1)
        ])
        new_img3 = transform(img_pil)
        plt.subplot(1,3,3)
        plt.imshow(new_img3)
        #new_img3.show()


        plt.show()

    def accuracyLoss(self):
        plt.imshow(mpimg.imread('plot.png'))
        plt.show()


    def inference(self):
        # load the image
        img = load_img(self.path, target_size=(32, 32))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 3 channels
        img = img.reshape(1, 32, 32, 3).transpose(0,3,1,2)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0


        images = img    
        print(images.shape)#torch.Size([1, 3, 32, 32])
        


        device = torch.device('cpu')
        net = VGG19()
        net.load_state_dict(torch.load('pretrainedModel.pth', map_location=device))

        images = torch.from_numpy(images)
        print(type(images))
        outputs = net(images)
        

        classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = images.clone().detach()
        img = img.reshape(len(img), 3, 32, 32).permute(0,2,3,1)#.astype("uint8")


  
        plt.subplot()          
        plt.imshow(img[0].numpy())

        plt.show()
        
        
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%s' % classes[predicted] ) , '\n ')
        
        out=outputs.detach().numpy()
        outlist = out.flatten()

        a ='Predicted: ', ' '.join('%s' % classes[predicted] ) , '  confidence:',  ' '.join('%f' % outlist[predicted] )

        plt.title(a)
        




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())