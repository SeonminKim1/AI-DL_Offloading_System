import os, cv2
import numpy as np
import random

import tensorflow as tf

IMAGE_WEIGHT, IMAGE_HEIGHT = (224,224)
input_shape=(IMAGE_WEIGHT, IMAGE_HEIGHT, 3)

standard_point = 800

path = 'D:/dataset/dogs-vs-cats/train1000'
save_dir=os.path.join(os.getcwd() + '\\by_keras') # 저장 경로 + 현재경로
model_name = 'vggnet_dogcat_model.h5'

# (1) img 기본 path설정
img_list = os.listdir(path)
random.shuffle(img_list)
print(img_list)
print('시작')

# (2) path 설정한 image 불러들이기.
set_size = len(img_list)
X_set = np.empty((set_size, IMAGE_WEIGHT,IMAGE_HEIGHT,3), dtype=np.uint8)#, dtype=np.float32)

# 1. 데이터 로드 및 image resize
for i, f in enumerate(img_list):
    img = cv2.imread(path + '/' + f)
    img = cv2.resize(img, (224,224))#.astype(np.float32)
    data = np.array(img)
    X_set[i] = data

# train, test 나누기.
X_train, X_test = X_set[0:standard_point], X_set[standard_point:set_size]

# 형태 출력해보기
print('X_train, X_test shape : ', X_train.shape, X_test.shape)
print('\nDone')

def representative_dataset_gen():
    for i in range(100):
        yield[X_train[i, None].astype(np.float32)]

# Quantize
converter = tf.lite.TFLiteConverter.from_keras_model_file('./by_keras/'+model_name) # input_shapes={"foo":[5,227,227,3]})
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen # data가 어떤 데이턴지 input형식등 알려주기 위함
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.uint8
converter.inference_output_type=tf.uint8
input_arrays=converter.get_input_arrays()
flat_data = converter.convert()

with open(os.getcwd() + '\\by_keras\\vggnet_dogcat_model.tflite', 'wb') as f:
    f.write(flat_data)
    print('tflite 파일 생성 완료')