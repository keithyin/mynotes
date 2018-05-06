# 使用virtualenv装tensorflow
如果你有两个项目，一个需要`python2.7`开发，一个需要`python3.5`开发，那么`virtualenv`是一个很好的选择。
## 准备
如果你想创建一个`python2.7`和`python3.5`的虚拟环境的话。首先你的电脑上得装有`python2.7`和`python3.5`，而且需要装好`pip`和`pip3`。安装`tf-gpu`版的话，确保你已经安装了`驱动`和`cuda`
## 安装
```shell
sudo pip install virtualenv #使用pip或pip3都可以，没啥影响
#创建环境，选择你想要的python版本（前提是你的电脑上已经安装好了）
virtualenv --no-site-packages --python=python3.5 ~/tensorflow

#激活环境
cd ~/tensorflow
source bin/activate
#安装tf,前面千万不要加sudo，执行任何pip命令都不要加sudo
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
#退出环境
deactivate
```
## pycharm 与 virtualenv
出现创建项目对话框，点击锯齿形状的按钮，选择`create virtualenv`，然后在`location`中选择你的`virtualenv`目录，就`OK`了
