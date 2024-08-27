from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa


# 影像大小
IMAGE_SIZE = (224, 224)

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 16

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 10

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-loss.h5'

# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory( 'training_dataset',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='binary',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory('validation_dataset',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='binary',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
"""
Found 16200 images belonging to 2 classes.
Found 1800 images belonging to 2 classes.
Class #0 = Cat
Class #1 = Dog
"""
# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
#output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
output_layer = Dense(1, activation='sigmoid')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
#net_final.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy', metrics=['accuracy'])
#1st loss function
#loss_fumction = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.34,gamma=2.0)
#2nd loss function
loss_fumction = tf.keras.losses.BinaryCrossentropy()
net_final.compile(optimizer=Adam(lr=1e-5),loss=loss_fumction, metrics=['accuracy'])

# 輸出整個網路結構
#print(net_final.summary())

# 訓練模型  fit_generator
history = net_final.fit(train_batches,
                steps_per_epoch = train_batches.samples // BATCH_SIZE,
                validation_data = valid_batches,
                validation_steps = valid_batches.samples // BATCH_SIZE,
                epochs = NUM_EPOCHS)

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)

# write result
fp = open("record.txt", "w")
fp.writelines("epoch\t loss\t accuracy\t val_loss\t val_accuracy\t\n")
for i in range(NUM_EPOCHS):
    fp.writelines("{}\t {:.4f}\t {:.4f}\t\t  {:.4f}\t t {:.4f}\t\n".format( 
        i+1, history.history['loss'][i], history.history['accuracy'][i], history.history['val_loss'][i], history.history['val_accuracy'][i]))
fp.close()

print("accurancy={}".format(history.history['accuracy'][-1]))

# plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss.png')
plt.show()

# plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc.png')
plt.show()