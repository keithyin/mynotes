# mxnet image I/O

本博客主要介绍 `mxnet` 如何读取读取图片进行训练



## RecordIO（mx.recordio）

mxnet 的 RecordIO 允许我们先将数据序列化成 `.rec` 文件，然后再从 `.rec` 文件中将序列化的文件解码出来。

**向 `.rec` 文件中写数据**

```python
record = mx.recordio.MXRecordIO('tmp.rec', 'w')
# 每次写入一条数据
for i in range(5):
    record.write('record_%d'%i)
record.close()
```

**从 `.rec` 文件中读数据**

```python
record = mx.recordio.MXRecordIO('tmp.rec', 'r')
while True:
    item = record.read() # read() 每次读一条数据
    if not item:
        break
    print (item)
record.close()
```



**当然，也可以对 record 加上索引，这就用到 IndexedRecordIO**

```python
# 向 IndexedRecordIO 中写数据
# 搞出了两个文件，一个存索引，一个存值
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
for i in range(5):
    record.write_idx(i, 'record_%d'%i)# 注意多了一个参数，那个参数代表 index
record.close()
```

```python
# 从 IndexedRecordIO 中读数据
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3) # 通过 idx 来读取
record.keys # 这个方法可以列出 文件中的所有 key
```



**`.rec` 文件中的每个 record 可以看作一个样本，每个 record可以是任意大小的 二进制数据**

但是，DL 中一般任务，样本都是 `data/label` pair，如何将这两个数据打包成一个 record 呢，请继续向下看。。。

`mx.recordio` 包中提供了一些工具函数用来完成这个工作，它们分别是 `pack, unpack, pack_img, unpack_img`



## pack 成一个 record，将一个 record unpack

**pack 和 unpack 用在 保存 float( 或 float array) label 和 二进制 data**

注意哦， pack用在保存：

* float [array] label
* 二进制 data




**data 和 一个 header** pack 到一起，组成了**一个 record**

什么是 header 呢？简单来说，就是用来保存 label 的。下面是 定义：

```python
mx.recordio.IRHeader(flag, label, id, id2)

# flat(int): 随便设值，可能用于啥骚操作
# label (float[array]): 一般用来 保存 data 的标签
# id(int): 一般是一个唯一的 id，用来表示 record
# id2(int): Higher order bits of the unique id, should be set to 0 (in most cases).
```

**header 与  data 一起，就成了 一个 data-label record了**



```python
# pack
data = b'data'
label1 = 1.0
header1 = mx.recordio.IRHeader(flag=0, label=label1, id=1, id2=0)
s1 = mx.recordio.pack(header1, data) # s1 是 二进制文件, data是 字节流 bytestring！

label2 = [1.0, 2.0, 3.0]
header2 = mx.recordio.IRHeader(flag=3, label=label2, id=2, id2=0)
s2 = mx.recordio.pack(header2, data)

# 可以将 s1, s2 写到 RecordIO 文件中了，RecordIO 和 IndexedRecordIO 都可以
```



```python
# unpack 将 record 中的文件 解码出来
mx.recordio.unpack(s1) # 返回一个 tuple，第一个元素是 Header，第二个是 二进制的 data
```



**如果数据是 image 的话， pack_img 就派上用场了**

```python
# pack
data = np.ones((3,3,1), dtype=np.uint8)
label = 1.0
header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
s = mx.recordio.pack_img(header, data, quality=100, img_fmt='.jpg')

# unpack
print(mx.recordio.unpack_img(s))
```



**贴心的 mxnet 提供了一个 工具函数用来将 img 转成 rec tools/img2rec.py, 在**[stc/tools](https://github.com/apache/incubator-mxnet/tree/master/tools)



现在问题是，如果 data label 都是图片，该怎么办呢？。。（遗留问题）



## Image IO

在 mxnet 中，有三种 图片预处理的方法：

* 使用 `mx.io.ImageRecordIter` ,特点： 速度快，不灵活，对于 目标识别很好使，但是 detection 和 segmentation 任务上就没法用了。
* 使用 `mx.recordio.unpack_img`(或者，cv2.imread(), skimage等)+numpy：特点，灵活，但是由于 pyython GIL 的存在，相当慢
* 使用 `mxnet` 提供的 `mx.image` 包。它将 image 存成 NDArray 类型，然后用 mxnet 的引擎来自动并行处理。




**使用[ImageRecordIter](http://mxnet.io/api/python/io.html#mxnet.io.ImageRecordIter)**

```python
data_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/caltech.rec", # the target record file
    data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options as defined in ImageRecordIter.
    )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```

**使用 [ImageIter](http://mxnet.io/api/python/image.html#mxnet.image.ImageIter)**

```python
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 227, 227),
                              path_imgrec="./data/caltech.rec",
                              path_imgidx="./data/caltech.idx" )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```



**其它**

```python
# mx.image 模块加载加载图片
b_img = open(path, mode='rb')
img = mx.image.imdecode(b_img.read()) # 将 string 或 byte string 解码成 NDArray
```









## 参考资料

[http://mxnet.io/tutorials/basic/data.html](http://mxnet.io/tutorials/basic/data.html)

[http://mxnet.io/api/python/io.html#mxnet.recordio.IRHeader](http://mxnet.io/api/python/io.html#mxnet.recordio.IRHeader)






