from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

test_datagen = ImageDataGenerator()#rescale=1.0/255)
test_batches = test_datagen.flow_from_directory('validation_dataset',
                                                target_size=(224, 224),
                                                interpolation='bicubic',
                                                class_mode='binary',
                                                shuffle=False,
                                                batch_size=16)

print(type(test_batches))

for batch in test_batches:
        x, y = batch
        print(y)
