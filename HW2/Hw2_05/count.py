#problem 5.1

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from PIL import Image

imageCat = 0
for filename in glob.glob(os.path.join("training_dataset\Cat", "*.jpg")):
    imageCat = imageCat +1
#for filename in glob.glob(os.path.join("validation_dataset\Cat", "*.jpg")):
#    imageCat = imageCat +1

imageDog = 0
for filename in glob.glob(os.path.join("training_dataset\Dog", "*.jpg")):
    imageDog = imageDog +1
#for filename in glob.glob(os.path.join("validation_dataset\Dog", "*.jpg")):
#    imageDog = imageDog +1
        
x = ["Cat", "Dog"]
y = [imageCat, imageDog]
plt.bar(x, y)

#set_xticks(x)
for i in range(len(x)):
    plt.text(i,y[i],y[i])

plt.title("Class Distribution")
plt.ylabel("Numbers of images")


plt.savefig("count.png")
plt.show()
