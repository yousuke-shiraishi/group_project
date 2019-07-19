import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import os


def load_data_training_and_test(datasetname):
    
    npzfile = np.load(datasetname + "_training_data.npz")
    train = npzfile["arr_0"]
    
    npzfile = np.load(datasetname + "_training_labels.npz")
    train_labels = npzfile["arr_0"]
    
    npzfile = np.load(datasetname + "_test_data.npz")
    test = npzfile["arr_0"]
    
    npzfile = np.load(datasetname + "_test_labels.npz")
    test_labels = npzfile["arr_0"]
    
    return (train, train_labels), (test, test_labels)




(X_train, y_train), (X_test, y_test) = load_data_training_and_test("wil_vs_cof")

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255




batch_size = 16
epochs = 8

img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]
input_size = (224, 224, 3)

"""
model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_size))
model.add(Activation("relu"))
model.add(Conv2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode=("same")))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(3))
model.add(Activation('sigmoid'))

print('---------------------------')

""""""

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = input_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

"""


from keras.applications.mobilenet_v2 import MobileNetV2

conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_size)

model = Sequential()
model.add(conv_base)

"""
model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Flatten())


model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))
"""

model.add(Flatten())

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))


set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block_14_expand':
        # 一切の学習を行わない場合
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

    
"""
for layer in conv_base__.layers:
    print(layer.name, ':', layer.trainable)
"""

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'rmsprop',
             metrics = ['accuracy'])


# どこまで実行しているか不明になるので...
from datetime import datetime
print('--------------------------------------------------------------------------------------')
print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))




model.summary()

# どこまで実行しているか不明になるので...
from datetime import datetime
print('--------------------------------------------------------------------------------------')
print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))




history = model.fit(X_train, y_train,
                   batch_size = batch_size,
                   epochs = 5,
                   validation_data = (X_test, y_test),
                   shuffle = True)

# 再構築可能なモデルの構造
# モデルの重み
# 学習時の設定 (loss，optimizer)
# optimizerの状態．これにより，学習を終えた時点から正確に学習を再開できます

model.save("multi_label.h5")
scores = model.evaluate(X_test, y_test, verbose=1)

print('Test loss', scores[0], 'Test accuracy', scores[1])

# どこまで実行しているか不明になるので...
from datetime import datetime
print('--------------------------------------------------------------------------------------')
print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))