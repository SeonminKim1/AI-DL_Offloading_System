################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
################################################################################
# %% NEED TO PROVIDE DATA AS EXAMPLE INPUT FOR CONVERSION/QUANTIFICATION
################################################################################

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

##### CONVERT DIMENSIONS FOR CONV2D
X_train = (X_train[:,:,:,np.newaxis]/255)
X_test = (X_test[:,:,:,np.newaxis]/255)
################################################################################
# %% CONVERT
################################################################################

##### GENERATOR FOR SAMPLE INPUT DATA TO QUANTIZE ON
def representative_dataset_gen():
    for i in range(100):
        yield [X_train[i, None].astype(np.float32)]

##### CREATE CONVERTER
#converter = tf.lite.TFLiteConverter.from_keras_model(model) # <-- ISSUES GETTING QUANTIZED!
converter = tf.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')

##### SHOW MODEL WHAT DATA WILL LOOK LIKE

##### QUANTIZE INTERNALS TO UINT8
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

##### SHOW MODEL WHAT DATA WILL LOOK LIKE
converter.representative_dataset = representative_dataset_gen

##### REDUCE ALL INTERNAL OPERATIONS TO UNIT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
#converter.inference_type = tf.uint8
input_arrays = converter.get_input_arrays()
print(input_arrays)
#converter.quantized_input_stats = {input_arrays[0]:(0., 1.)}
##### CONVERT THE MODEL
tflite_model = converter.convert()

##### SAVE MODEL TO FILE
tflite_model_name = "mnist.tflite"
open(tflite_model_name, "wb").write(tflite_model)

##### MODEL SHOULD NOW BE COMPILED!
# : edgetpu_compiler -s mnist.tflite

################################################################################
# %% VARIOUS OPTIONS FOR EXPORTING...
################################################################################

#converter.representative_dataset = representative_dataset_gen
#converter.input_shapes= (1,28,28,1)
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#converter.inference_type = tf.float32
#converter.std_dev_values = 0.3
#converter.mean_values = 0.5
#converter.default_ranges_min = 0.0
#converter.default_ranges_max = 1.0
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                       tf.lite.OpsSet.SELECT_TF_OPS]
#converter.post_training_quantize=True
#    --input_arrays=conv2d_input \
#    --output_arrays=dense/Softmax \

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
