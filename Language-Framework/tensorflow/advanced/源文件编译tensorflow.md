# 源码编译 tensorflow cpu 总结(ubuntu16.04)



## 安装依赖

```shell
# 如果编译 python2， 安装以下依赖
sudo apt-get install python-numpy python-dev python-pip python-wheel 

# 如果编译 python3, 安装以下依赖
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel

# 如果编译 GPU 版本， 先安装CUDA，然后安装。这里只介绍编译cpu版本。因为我没有测试gpu的编译
sudo apt-get install libcupti-dev 

# 安装python 依赖
sudo pip install six numpy wheel 
```

**安装 bazel：**

* 下载地址 [bazel](https://github.com/bazelbuild/bazel/releases), 我安装的是 0.5.1-without-jdk-installer-linux-x86_64.sh 版本。因为我之前已经安装好jdk了
* 下载完成之后，执行 .sh 文件安装

## 下载与编译

```shell
# 从 git 上下载 tensorflow
git clone https://github.com/tensorflow/tensorflow

cd tensorflow
git checkout r1.1 # 执行这个指令如果有错误的话，需要先git add. , git commit -m "desc" 一下。

# 安装配置，按照自己的需求选择
./configure

# 构建 pip package， cpu， 过程中会出现一些 warning，不用理会。
# 生成的 pip 包 在  /tmp/tensorflow_pkg 文件夹中。
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# 安装 pip package
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.1.0-py2-none-any.whl
```



## 测试

```python

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(tf.__version__)
print(sess.run(hello))
```





## 参考资料

[https://www.tensorflow.org/install/install_sources](https://www.tensorflow.org/install/install_sources)