# オーグメンてーしょんしたい画像が入ったディレクトリ
my_path = 'image_folder/images/'



from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm



def augmentation(dir_path, initial_letter_of_file='w', augment_num=2):
    
    
    """
    note : 指定ディレクトリ内の,指定頭文字で始まるファイルを指定枚数オーグメントする
    ----------
    dir_path : フォルダパス
    initial_letter : augmentしたいファイル名の頭文字
    aument_num : augmentしたい枚数
    ----------
    """
    
    
    files_name = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    files_name.remove('.DS_Store')
    
    
    datagen = ImageDataGenerator(rotation_range=40,
                             #width_shift_range=0.2,
                             #height_shift_range=0.2,
                             #shear_range=0.2,
                             #zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest'
    )
    
    
    for i, file in tqdm(enumerate(files_name)):
        img = load_img(dir_path + file)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) 

        if file[0] == initial_letter_of_file:
            i = 0
            for batch in datagen.flow(x, save_to_dir=dir_path, save_prefix=initial_letter_of_file, save_format="jpg"):
                i += 1
                if i > augment_num:
                    break

        else:
            pass
        
        
        
        

# 関数の実行
augmentation(my_path, 'd')
augmentation(my_path, 'c')





import cv2
import numpy as np
import sys
import shutil
from tqdm import tqdm


def train_test_split(files_name, class_0='d', class_1='c', train_size=0.8):
    
    """
    note : 画像フォルダから、指定クラスを、指定割合でtrain_sprit
    ----------
    class_0 : クラス名（今回はwilkinson）
    class_1 : クラス名（今回はcoffee）
    train_size : 分割したい割合
    ----------
    """
    
    class_0_count = 0
    class_1_count = 0
    
    each_class_size = len(files_name) // 2
    
    train_size = each_class_size * train_size
    test_size = each_class_size - train_size
    
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []
    training_file_name = []
    test_file_name = []
    
    size=224
    
    for i, file in tqdm(enumerate(files_name)):
        
        if files_name[i][0] == class_0:
            class_0_count += 1
            image = cv2.imread(my_path + file)
            image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
            if class_0_count <= train_size:
                training_images.append(image)
                training_labels.append(0)
                training_file_name.append(file)
                cv2.imwrite(class0_dir_train + class_0 + str(class_0_count) + '.jpg', image)
            if class_0_count > train_size and class_0_count <= train_size + test_size:
                test_images.append(image)
                test_labels.append(0)
                test_file_name.append(file)
                cv2.imwrite(class0_dir_val + class_0 + str(class_0_count) + '_' + '.jpg', image)

        if files_name[i][0] == class_1:
            class_1_count += 1
            image = cv2.imread(my_path + file)
            image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
            if class_1_count <= train_size:
                training_images.append(image)
                training_labels.append(1)
                training_file_name.append(file)
                cv2.imwrite(class1_dir_train + class_1 + str(class_1_count) + '.jpg', image)
            if class_1_count > train_size and class_1_count <= train_size + test_size:
                test_images.append(image)
                test_labels.append(1)
                test_file_name.append(file)
                cv2.imwrite(class1_dir_val + class_1 + str(class_1_count) + '_' + '.jpg', image)
                
    return training_images, training_labels, test_images, test_labels, training_file_name, test_file_name



# 関数の実行
training_images, training_labels, test_images, test_labels, training_file_name, test_file_name = train_test_split(updated_files_name)



# 保存用ディレクトリの作成
# オーグメンテーションした画像をクラスごとにtrain, test用に格納していく

import os
import shutil

class0_dir_train = 'image_folder/train/class0/'
class0_dir_val = 'image_folder/test/class0/'
class1_dir_train = 'image_folder/train/class1/'
class1_dir_val = 'image_folder/test/class1/'

def make_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    
make_dir(class0_dir_train)
make_dir(class0_dir_val)
make_dir(class1_dir_train)
make_dir(class1_dir_val)





np.savez('wil_vs_cof_training_data.npz', np.array(training_images))
np.savez('wil_vs_cof_training_labels.npz', np.array(training_labels))
np.savez('wil_vs_cof_test_data.npz', np.array(test_images))
np.savez('wil_vs_cof_test_labels.npz', np.array(test_labels))





