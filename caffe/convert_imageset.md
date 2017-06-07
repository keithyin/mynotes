# convert_imageset



```shell
 convert_imageset [FLAGS] file_root_folder file_list db_name
```

* FLAGS:
  - --gray: 是否以灰度图的方式打开图片。程序调用opencv库中的imread()函数来打开图片，默认为false
  - --shuffle: 是否随机打乱图片顺序。默认为false
  - --backend:需要转换成的db文件格式，可选为leveldb或lmdb,默认为lmdb
  - --resize_width/resize_height: 改变图片的大小。在运行中，要求所有图片的尺寸一致，因此需要改变图片大小。 程序调用opencv库的resize（）函数来对图片放大缩小，默认为0，不改变
  - --check_size: 检查所有的数据是否有相同的尺寸。默认为false,不检查
  - --encoded: 是否将原图片编码放入最终的数据中，默认为false
  - --encode_type: 与前一个参数对应，将图片编码为哪一个格式：‘png','jpg'......
* file_root_folder： 图片文件的存放路径
* file_list ： txt文件的存放路径
* db_name: 生成的DB文件的存放路径（dir）



# compute_image_mean

```shell
compute_image_mean db_name output_proto
```

* db_name : db数据库的名字
* output_proto: 输出文件的名字