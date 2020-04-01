import os, cv2
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

input_shape=(227, 227, 3)
batch_size=32
num_classes=2
epoch=1
standard_point = 800


path = 'C:/Users/urse/Desktop/dataset/dogs-vs-cats/train1000'
save_dir=os.path.join(os.getcwd() + '\\by_keras') # 저장 경로 + 현재경로
model_name = 'alexnet_dogcat_model.h5'

# (1) img 기본 path설정
img_list = os.listdir(path)
random.shuffle(img_list)
print(img_list)
print('시작')
# (2) path 설정한 image 불러들이기.
set_size = len(img_list)
X_set = np.empty((set_size, 227,227,3), dtype=np.uint8)#, dtype=np.float32)
y_set = np.empty((set_size), dtype=np.uint8)

# 1. 데이터 로드 및 image resize
with open('dogcat_trainlabel.csv', 'w') as csv1:
    with open('dogcat_testlabel.csv', 'w') as csv2:
        for i, f in enumerate(img_list):
            if i%10 ==0:
                print("Reading subset data: {}/{}...".format(i,set_size), end='\r')
            label = f.split('.')[0]
            if label=='cat':
                y=0
            else: # label =='dog'
                y=1
            img = cv2.imread(path + '/' + f)
            img = cv2.resize(img, (227,227))#.astype(np.float32)
            data = np.array(img)
            X_set[i] = data
            y_set[i] = y
            if i<standard_point: # 800장 이하이면
                csv1.write(label + ',' + str(y) + '\n') # trainlabel
            else:
                csv2.write(label + ',' + str(y) + '\n') # testlabel

y_set_onehot = np.zeros((set_size, 2), dtype=np.uint8)
y_set_onehot[np.arange(set_size), y_set] =1
y_set = y_set_onehot

# train, test 나누기.
X_train, X_test = X_set[0:standard_point], X_set[standard_point:set_size]
Y_train, Y_test = y_set[0:standard_point], y_set[standard_point:set_size]

# testset 을 npy와 label로 저장하기.
np.save('dogcat_trainset.npy', X_train)
np.save('dogcat_testset.npy', X_test)

# 형태 출력해보기
print('X_train, X_test shape : ', X_train.shape, X_test.shape)
print('Y_train, Y_test shape : ', Y_train.shape, Y_test.shape)
print('\nDone')

# Create the model
model=Sequential()

# Layer1
model.add(Conv2D(96, (11, 11), strides=4,
                 padding='same', input_shape=input_shape))

# Layer2
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Layer3
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Layer4
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))

# Layer5
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Layer 6 ~ 8 fully-connected Layer
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
model.summary()
model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epoch, shuffle=True)
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved Trained Model %s' %model_path)

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test Accuracy:', scores[1])
