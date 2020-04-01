# Google_Coral_Edgetpu_USBAccelerator
### 구글 Coral USB Accelerator & Raspiberry Pi 3 B+
### 1. python3 mnist_make_model.py 실행 -> keras_model.h5 생성
### 2. python3 mnist_convert_tflite.py 실행 -> mnist.tflite 생성
### 3. edgetpu_compiler -s mnist.tflite -> mnist_edgetpu.tflite 생성됨.
### 4. 라즈베리파이서 진행 
- workon coral 실행
- python3 classify_mnist.py --model=mnist_edgetpu.tflite --image=mnist_new_data/0_003.jpg
- python3 classify_mnist.py --model=mnist_edgetpu.tflite --image=mnist_new_data/3_018.png