import tensorflow as tf
img = tf.placeholder(name="img", dtype=tf.float32, shape=(1,64,64,3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.fake_quant_with_min_max_args(val, min=0., max=1., name="output")

with tf.Session() as sess:
	converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
	converter.inference_type=tf.uint8
	input_arrays=converter.get_input_arrays()
	converter.quantized_input_stats={input_arrays[0] : (0., 1.)} # mean std_dev
	tflite_model = converter.convert()
	open("converted_model_quant.tflite", "wb").write(tflite_model)
