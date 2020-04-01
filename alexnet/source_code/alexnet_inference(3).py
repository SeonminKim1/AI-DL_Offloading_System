
import argparse
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from edgetpu.basic.basic_engine import BasicEngine
import edgetpu.classification.engine

from PIL import Image
from datetime import datetime
import time
import numpy as np
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=False)
  args = parser.parse_args()

  # Prepare labels.
  labels = dataset_utils.read_label_file(args.label)
  # Initialize engine.
  engine = BasicEngine(args.model)

  # Run inference.
  testImages = np.load('dogcat_testset.npy')
  img = testImages.flatten()
  result = engine.run_inference(img)
  #for result in engine.classify_with_image(img, top_k=1):
  print('---------------------------')
  print('Input Tensor: ', engine.get_input_tensor_shape())
  print('Output Tensor: ', result[1])
  print('Score: ', np.argmax(result[1]))

if __name__ == '__main__':
#  print('EdgeTPU RuntimeVersion:', edgetpu.basic.edgetpu_utils.GetRuntimeVersion())
  sum=0
  for i in range(1, 11):
    start_time = time.time()
    s_time = datetime.today()
    main() #
    e_time = datetime.today()
    end_time = time.time()
    during_time = round(end_time-start_time, 2)
    print('No.',i, 'Start Time ', s_time)
    print('No.',i, 'End Time: ', e_time)
    print('No.',i, 'During Time: ', during_time)
    sum = sum + during_time
  print('Average During Time : ', round((sum/i),2))