

import argparse
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image


# 실제 이미지파일을 로드해서 사용하는 방법.
# 입력명령어 python3 vggnet_dogcat_real_image_inference.py --model=vggnet_dogcat_model_edgetpu.tflite --link=test_dat --count=9

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.', required=False)
  parser.add_argument(
      '--link', help='Folder name to be recognized. ex)test_data', required=True)
  parser.add_argument(
      '--count', help='How many times do you want', required=True)
  args = parser.parse_args()

  # Prepare labels.
  # dictionary 형태로 반환 (1:'dog,1', 2:'cat,0', 3:'dog,1')

  #labels = dataset_utils.read_label_file(args.label)
  # Initialize engine.
  engine = ClassificationEngine(args.model)

  # Run inference.
  for i in range(1, int(args.count)+1):
      path = args.link
      img = Image.open(path+'/'+str(i)+'.jpg')
      resize_img = img.resize((224,224))

      # classify_with_image값으로 list[int, float]리턴. 추론정답label과, Confidence(신뢰도)임.
      for result in engine.classify_with_image(resize_img, top_k=3):
        print('---------------------------')
        print('Inference Answer Label', result[0])
        print('Score : ', result[1])

if __name__ == '__main__':
  main()
