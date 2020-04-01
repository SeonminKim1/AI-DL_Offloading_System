"""A demo to classify image."""
import argparse
from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  #parser.add_argument(
  #    '--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=True)
  args = parser.parse_args()

  # Prepare labels.
  #labels = ReadLabelFile(args.label)
  # Initialize engine.
  engine = BasicEngine(args.model)
  # Run inference.
  array_size = engine.required_input_array_size()

  image = Image.open(args.image)
  print(image)
  pix = np.array(image) # PILImage를 numpy 배열로 바꾸는 법
  img = pix.flatten() # PILImage를 numpy 배열로 바꿈.
  print('\n------------------------------')
  print('Run infrerence.')
  print('  required_input_array_size: ', array_size)
#  print('  input shape: ', img.shape)

  result = engine.run_inference(img)
  print('------------------------------')
  print('Result.')
  print('Inference latency: ', result[0], ' milliseconds')
  print('Output tensor: ', result[1])
  print('Inference:(argmax of output tensor) ', np.argmax(result[1]))

if __name__ == '__main__':
  main()
