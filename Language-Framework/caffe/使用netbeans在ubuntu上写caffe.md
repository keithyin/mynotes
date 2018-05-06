# 使用C++写caffe代码,如何配置

**环境:**

* ubuntu
* g++5.4
* netbeans
* caffe



**环境变量配置：**

* `LD_LIBRARY_PATH` 在`/etc/profile`中配置这个环境变量
  * 包括 `caffe/build/lib; opencv/lib; opencv/share/OpenCV/3rdparty/lib`
* LIBRARY 也要配置好
  * `caffe/build/lib`



**netbeans 项目配置：**

* `右击项目名`->`属性`->`C++ complier`->`include Directories`->`caffe/include;/usr/local/include`
* `右击项目名`->`属性`->`Linker`->`Libraries`-> `Add Library`->`boost_system`
* `右击项目名`->`属性`->`Linker`->`Libraries`-> `Add Library`->caffe/build/lib/...so 文件