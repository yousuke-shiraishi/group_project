import cv2
from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join
from keras.models import model_from_json
import time


class predict_class:

    def __init__(self, model='multi_label.h5', test_image_path='test_image/'):
        """
        attribute
        ----------------
        model : model(.h5)
        test_image_path : テストファイルが入っているディレクトリ
        """
        self.model = model_from_json(open('../models/bottle_model_2.json').read())
        self.model.load_weights(model)
        #self.model = load_model(model)
        self.test_image_path = test_image_path


    def predict(self, image, class_0='soda_float', class_1='fanta_litchi', class_2='cclemon', class_3='namacha', class_4='cocacola'):
        """
        note : 与えられたimageのpathから、多値分類を行う関数
        ----------------
        attribute
        image : 入力された画像のnumpy配列
        ----------------
        """
        class_name = [class_0, class_1, class_2, class_3, class_4]
        image = cv2.resize(image, (224, 224))
        sample_arr = np.expand_dims(image, axis=0)
        sample_arr = sample_arr /255
        result = self.model.predict(sample_arr)
        return class_name[np.argmax(result)], np.max(result)


    def accuracy_check(self):
        """
        note : test_imageディレクトリ内の複数のテストデータで、モデルの精度判定する関数
        ---------------
        """
        print('--------------------------------------------------------------------------------------')
        # 時間計測（start）
        start = time.time()

        # 画像ファイルの取り出し
        test_images = [f for f in listdir(self.test_image_path) if isfile(join(self.test_image_path, f))]
        test_images.remove('.DS_Store')

        collect = 0
        for test_image in test_images:
            class_name = self.predict(self.test_image_path + test_image)
            print('file name :', test_image, '/ class name :', class_name)
            if test_image[0] == class_name[0]:
                print('○')
                collect += 1
            else:
                print('×')
            print('--------------------------------------------------------------------------------------')


        print('正解数は{sample}サンプル中{collect}で、testデータの正解率は{rate}です。'.format(sample=len(test_images),
                                                                        collect=collect,
                                                                        rate=collect/len(test_images)))
        print('--------------------------------------------------------------------------------------')

        # 時間計測（end）
        elapsed_time = time.time() - start
        #print('合計秒数:：',elapsed_time)
        print('画像ごとの処理時間(秒) :', str((elapsed_time)/len(test_images)) )

        # どこまで実行しているか不明になるので...
        print('--------------------------------------------------------------------------------------')
        print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    def output_logit(self, image, layer_name='dense_12'):
        """
        note : 特定層においてロジットの出力をarrayにて出力する関数
        ----------
        attribute
        layer_name : layer
        image : 入力された画像のnumpy配列
        """

        hidden_layer_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output)

        image = cv2.resize(image, (224, 224))
        target = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])).astype('float') / 255.0
        array = hidden_layer_model.predict(target)
        std = np.std(array)
        var = np.var(array)

        return array, std, var


'''
if __name__ == '__main__':
    # インスタンスの作成
    predict = predict_class()

    # テストファイルのパスフォルダ名
    test_image_path = 'test_image/'

    predict.predict(test_image_path + 'aquarius__20190709160943.jpg')
    predict.accuracy_check()

# どこまで実行しているか不明になるので...
#from datetime import datetime
#print('--------------------------------------------------------------------------------------')
#print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
'''
