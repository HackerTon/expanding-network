import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

graph = tf.Graph()

with graph.as_default():
    with tf.Session() as sess:
        tensor_graph = trt.create_inference_graph()