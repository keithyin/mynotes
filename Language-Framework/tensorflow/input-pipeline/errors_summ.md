# 使用 tfrecords 时的常见错误总结



1. **确保 string_input_producer 中的文件名字是正确的。 **

2. `string_input_producer(file_names, num_epochs=100000, shuffle=True)` 当指定 num_epochs 时，在初始化模型参数的时候，一定要 记得 `tf.local_variables_initializer().run()` , 因为 tf 会将 num_epoch 作为 `local variable`。 否则的话，会报错 `Attempting to use uninitialized value ReadData/input_producer/limit_epochs/epochs`

3. **解码 tfrecords 时的类型一定要和制作 tfreords 时的类型一致：**

   这个问题主要出现在 bytestring 上，在保存图片数据时候，我们通常会

   1. 将图片  .tostring()  转成 bytestring
   2. 然后 再制作 tfrecords
   3. 然后在解码的时候，我们会用 decode_raw 将bytestring 解码出来。
   4. 然后再用 cast 转成目标数据类型。

   这里要注意的是，cast 转成的目标数据类型一定要和 .tostring() 之前的数据类型一致。



