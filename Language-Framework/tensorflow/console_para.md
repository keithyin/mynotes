全局环境下编写代码
```python
import tensorflow as tf
flags = tf.flags
logging = tf.logging
flags.DEFINE_string("para_name","default_val", "description")
flags.DEFINE_bool("para_name","default_val", "description")

FLAG = flags.FLAG
def main(_):
	FLAG.para_name

if __name__ = "__main__":
	tf.app.run() #解析命令行参数，调用main函数 main(sys.argv)
```