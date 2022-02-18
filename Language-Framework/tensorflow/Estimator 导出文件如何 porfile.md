```python
from __future__ import print_function

from tensorflow.saved_model import tag_constants
import tensorflow as tf
from tensorflow_core.python.client import timeline
from tensorflow.python.saved_model.loader_impl import SavedModelLoader
import tfrecord2placeholders


def do_infer(export_dir, tfrecord_file, batch_size):
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    loader = SavedModelLoader(export_dir)

    saver, _ = loader.load_graph(tf.get_default_graph(), tags=[tag_constants.SERVING])

    with tf.Session() as sess:
        loader.restore_variables(sess, saver)

        placeholders = [x.name for x in tf.get_default_graph().get_operations()
                        if x.type == "Placeholder"]
        ctr = tf.get_default_graph().get_tensor_by_name("ctr:0")
        for i, data in enumerate(tfrecord2placeholders.get_infer_data(tfrecord_file, batch_size)):
            feed_dict = {"{}:0".format(ph_name): data[ph_name] for ph_name in placeholders}

            res = sess.run(ctr, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            print(res)
            if (i + 1) % 10 == 0:
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open("timeline_{:04d}.json".format(i), "w") as f:
                    f.write(ctf)


if __name__ == '__main__':
    do_infer("saved_model", "push_content_rankin.data", 10)

```

打开 chrome, 地址栏输入 `chrome://tracing/`, 导入写出的文件看 timeline 即可。
