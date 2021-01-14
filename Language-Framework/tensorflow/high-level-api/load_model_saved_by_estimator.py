import tensorflow as tf
from tensorflow.saved_model import tag_constants
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_fake_input():
    # fea_desc = {"x": data}
    examples = []
    for _ in range(10):
        feature = {"x": _bytes_feature(np.random.randn(784).astype(np.float32).tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(example.SerializeToString())
    return examples


export_dir = "model_dir/export/best_exporter/1610332394"

with tf.Session(graph=tf.Graph()) as sess:
    # using "saved_model_cli show --dir exported_model_dir --all" to check the basic info of saved model
    # tag & tensor name can obtained from there.
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
    classes_tensor = tf.get_default_graph().get_tensor_by_name("ArgMax:0")
    prob_tensor = tf.get_default_graph().get_tensor_by_name("softmax_tensor:0")
    input_tensor = tf.get_default_graph().get_tensor_by_name("Placeholder:0")

    print(classes_tensor, prob_tensor, input_tensor)
    tf.get_default_graph().finalize()

    fake_inp = get_fake_input()
    print(sess.run([prob_tensor, classes_tensor], feed_dict={input_tensor: fake_inp}))
