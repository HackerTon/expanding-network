import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import argparse

# Debug
import numpy as np


class tensorTrt:
    def __init__(self, path_to_saved_model, batch_size, max_gpu_mem_size_for_trt):
        self.graph = tf.Graph()
        self.session = tf.Session()

        self.tensor_graph = trt.create_inference_graph(input_graph_def=None,
                                                       outputs=None,
                                                       input_saved_model_dir=path_to_saved_model,
                                                       input_saved_model_tags=['serve'],
                                                       max_workspace_size_bytes=max_gpu_mem_size_for_trt,
                                                       max_batch_size=batch_size,
                                                       precision_mode='INT8')

        self.output = tf.import_graph_def(graph_def=self.tensor_graph,
                                          return_elements=['final_dense_layer/Softmax:0'])

    def infer(self, input):
        return self.session.run(self.output, feed_dict={'import/input_1:0': input})


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('echo')
    # args = parser.parse_args()
    # print(args.echo)

    tensor_1 = tensorTrt(path_to_saved_model='savemodel',
                         batch_size=1,
                         max_gpu_mem_size_for_trt=1024)

    while True:
        input_array = np.random.rand(1, 244, 244, 3)

        print(tensor_1.infer(input_array))
