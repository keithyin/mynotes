# paddle 输入流水线





# Dataset 套件

* Dataset:
  * Dataset 会根据设置的线程数构建相应数量的 DataFeed(data_reader)
    * `DatasetImpl<T>::CreateReaders()`
  * DataFeed并发将数据读到灌到内存中
    * `DatasetImpl<T>::LoadIntoMemory()`
    * 搞进来的被一个 channel_writer 接收了...
* DataFeed, 也叫 data_reader
* 
* DataReader