
import argparse
from edgetpu.utils import dataset_utils
from edgetpu.basic.basic_engine import BasicEngine
import edgetpu.basic.edgetpu_utils

from PIL import Image
from datetime import datetime
import time
import numpy as np
import os

# Coral API Edgetpu.basic.basic_engine 함수들
# Edgetpu.utils.dataset -> read_label_file -> return dictionary 형태로
# python3 vggnet_inference(3).py --model=vggnet_dogcat_model_edgetpu.tflite --label=dogcat_testlabel.csv --count=199
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=False)
    parser.add_argument(
        '--count', help='How many times do you want', required=True)
    args = parser.parse_args()

    # Prepare labels.
    # dictionary 형태로 반환 (1:'dog,1', 2:'cat,0', 3:'dog,1')
    labels = dataset_utils.read_label_file(args.label)
    # Initialize engine.
    engine = BasicEngine(args.model)

    # Run inference.
    testImages = np.load('dogcat_testset.npy')
    sum, count = 0, 0
    for i in range(1 , int(args.count)+1):
        img = testImages[i].flatten()
        # result 는 튜플 / result[0] - inference_latency / result[1] -> array형태의 output_tensor
        result = engine.run_inference(img)
        print('---------------------------')
        print('Input Tensor: ', engine.get_input_tensor_shape()) # input_tensor_shape
        #print('required_input_array_size()', str(engine.required_input_array_size())) # 224x224x3
        #print('get_inference_time()', str(engine.get_inference_time()), 'ms') # 추론시간 time
        #print('get_num_of_output_tensors()', str(engine.get_num_of_output_tensors())) # output tensor 갯수
        #print('get_output_tensor_size(0)', str(engine.get_output_tensor_size(0))) # 텐서 여러개 반환시 해당텐서 번호
        #print('get_all_output_tensors_sizes()', str(engine.get_all_output_tensors_sizes())) # ouput 텐서 크기
        #print('total_output_array_size()', str(engine.total_output_array_size())) # output 텐서의 배열 크기
        print('Output Tensor: ', result[1])
        during_time = round(result[0]/1000,2)
        print('No.', i, 'Runtime:', str(during_time), 'sec')
        result_index = str(np.argmax(result[1])) # 추론으로 나온 정답
        answer_index = labels[i].split(',')[1] # 정답 text label에서 뽑아낸 것
        if result_index == answer_index:
            print(i, '번째는 정답!')
            count = count + 1
        else:
            print(i, '번쨰는 틀림!')
            #print('Score:', labels(result[0]))
        #print('Score: ', np.argmax(result[1]))
        sum = sum + during_time
        print('\n')
    print('Average During Time : ', round((sum / i), 2))
    print('Accuracy', round((count / i), 3))
if __name__ == '__main__':
    print('EdgeTPU RuntimeVersion:', edgetpu.basic.edgetpu_utils.GetRuntimeVersion())
    main() #
