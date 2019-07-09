import cv2
from keras.models import load_model

def predict_class(image, class_0='dog', class_1='cat'):
    """
    note : モデルをロードし、バイナリーで分類する関数
    ---------------
    """
    sample = load_img(image, target_size=(224,224))
    sample_arr = img_to_array(sample)
    sample_arr = np.expand_dims(sample_arr, axis=0)
    sample_arr = sample_arr /255
    model = load_model('wil_vs_cof.h5')
    result = model.predict(sample_arr)
    print('result prob', result)
    if result < 0.5:
        return class_0
    else:
        return class_1
    
    
def predict_multi_class(image, class_0='アクエリ', class_1='ソーダフロート', class_2='cc レモン', class_3='ファンタ'):
    """
    note : モデルをロードし、マルチラベルで分類する関数
    ---------------
    """
    class_name = [class_0, class_1, class_2, class_3]
    sample = load_img(image, target_size=(224,224))
    sample_arr = img_to_array(sample)
    sample_arr = np.expand_dims(sample_arr, axis=0)
    sample_arr = sample_arr /255
    model = load_model('multi_label.h5')
    result = model.predict(sample_arr)
    print('result prob', class_name[np.argmax(result)])
    
    
    
    
    
# test ファイルが格納されているディレクトり
test_image_path = 'test_image/'






from os import listdir
from os.path import isfile, join
import time


def accuracy_check(test_image_path):
    """
    note : test_imageディレクトリ内のテストデータで、モデルの精度判定する関数
    ---------------
    attribute
    test_image_path : test_imageディレクトリのパス
    """
    
    # 時間計測（start）
    start = time.time()
    
    # 画像ファイルの取り出し
    test_images = [f for f in listdir(test_image_path) if isfile(join(test_image_path, f))]
    test_images.remove('.DS_Store')

    collect = 0
    for test_image in test_images:
        class_name = predict_class(test_image_path + test_image)
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
from datetime import datetime
print('--------------------------------------------------------------------------------------')
print('This code was runned on date / time below', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))




accuracy_check(test_image_path)