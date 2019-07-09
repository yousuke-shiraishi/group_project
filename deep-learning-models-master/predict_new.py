import keras
import numpy as np
from keras.preprocessing import image
import time

img_height, img_width = 150, 150
model_dir = './model/'
model = keras.models.load_model(model_dir + 'model_InceptionV3.hdf5', compile = False)
time1 = time.time()
# filename = "./data/validation/coffee_img/coffee_img_0001.jpg"
filename = "./images.jpeg"
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
# これを忘れると結果がおかしくなるので注意
x = x / 255.0

# print(x)
# print(x.shape)

# クラスを予測
# 入力は1枚の画像なので[0]のみ
pred = model.predict(x)[0]
time2 = time.time()
print(pred)
print(time2 - time1)