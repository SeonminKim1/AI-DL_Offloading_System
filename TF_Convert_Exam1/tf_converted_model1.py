import tensorflow as tf
if not str(tf.__version__).startswith('1.15'):
    print('please use tensorflow 1.15')
    exit()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

tf.enable_eager_execution()

image_shape = (64,64,3)

# Creating a dummy keras model here
x = Input(shape=image_shape)
y = Conv2D(3, (3, 3), padding='same')(x)
model = Model(inputs=x, outputs=y)
model.summary()
model.save('keras_model.h5', include_optimizer=False)

def representative_dataset_gen():
    for i in range(100):
        # creating fake images
        image = tf.random.normal([1] + list(image_shape))
        yield [image]

# actual conversion
converter = tf.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen # tf.lite.RepresentativeDataset(representative_dataset_gen)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # For EdgeTPU, no float ops allowed
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# save model
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)
