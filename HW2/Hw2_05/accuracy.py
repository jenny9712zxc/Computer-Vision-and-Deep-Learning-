#problem 5.4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

test_datagen = ImageDataGenerator()#rescale=1.0/255)
test_batches = test_datagen.flow_from_directory('validation_dataset',
                                                target_size=(224, 224),
                                                interpolation='bicubic',
                                                class_mode='categorical',
                                                shuffle=True,
                                                batch_size=16)

net1 = load_model('model-resnet50-loss2.h5')
loss1, accuracy1 = net1.evaluate(test_batches)
print('accuracy={:.4f}'.format(accuracy1))

net2 = load_model('model-resnet50-loss1.h5')
loss2, accuracy2 = net2.evaluate(test_batches)
print('accuracy={:.4f}'.format(accuracy2))

x_label = ["Binary Cross Entropy", "Focal Loss"]
accuracy = [accuracy1, accuracy2]
x = np.arange(2)
for i in range(len(x)):
        plt.text(i,accuracy[i],accuracy[i])

#plt.text(x,y,text)

plt.bar(x, accuracy)

plt.title("Accuracy Comparsion")
plt.xticks(x, x_label)
plt.ylabel("accuracy")
plt.savefig("accuracy.png")
plt.show()
