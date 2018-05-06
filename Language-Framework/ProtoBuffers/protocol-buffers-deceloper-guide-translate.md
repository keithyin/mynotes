# Protocol Buffers Developer Guide

欢迎来到 `protocol buffers` 的开发者文档 - `protocol buffers`, 一个语言中立,平台中立, 用于通信协议,数据存储 等 序列化结构数据的可扩展方式.

## 什么是 protocol buffers ?
`Protocol Buffers` 是 序列化结构数据的一种 灵活的,有效的,自动的机制, 想想 `XML`,但是它比 `XML`更小,更快,更简单. 一旦定义好你的结构数据, 之后,你就可以使用 一种特殊的生成源码 从不同的数据流中, 使用不同的语言去读和写你的结构数据. 甚至,你可以在不打断已经部署好的程序情况下更新你的数据结构.

## 它是怎么工作的?
在 `.proto` 文件中,通过定义 `protocol buffer message` 类型来指定如何序列化你的结构数据. 每个 `protocol buffer message`都是一个小的逻辑单元,用与记录信息, 它包含了一系列的 `name-value` 对.下面是一个简单的`.proto`文件例子,它定义了一个`message`,用于包含人的信息.
```
message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;
  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }
  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }
  repeated PhoneNumber phone = 4;
}
```  
可以看到,`message`的格式是非常简单的--每个`message`类型都有一个或多个 编号的 字段, 每个字段都有一个名字和值的类型. 值的类型可以是数值类型(整型或者浮点型), 布尔型, 字符串, `raw bytes`, 甚至是 `protocol buffer message`类型.允许你去层次的结构化数据. 你可以指定 可选字段, [必填字段](https://developers.google.com/protocol-buffers/docs/proto#required_warning),和重复字段.在[这里](https://developers.google.com/protocol-buffers/docs/proto),你可以找到关于如何写 `.proto`文件更详细的信息.

一旦定义好了 `messages`, 你就可以  在`.proto`文件上运行 编程语言相关的`protocol buffer compiler`去生成数据访问类(data access classes). 它们对每个字段提供了简单的访问器(就像 `name()`, 和 `set_name()`) 和 将整个结构 从 `raw bytes`解析和序列化成`raw bytes`的方法. 例如, 如果选择使用 `c++` 语言, 对上述的例子进行编译会生成一个 `Person` 类. 之后,就可以在应用程序中使用这个类进行填充,序列化和检索 `Person protocol buffer messages`. 你可能会写出以下代码:
```c++
Person person;
person.set_name("John Doe");
person.set_id(1234);
person.set_email("jdoe@example.com");
fstream output("myfile", ios::out | ios::binary);
person.SerializeToOstream(&output);
```

之后,你可以将`message` 读入
```c++
fstream input("myfile", ios::in | ios::binary);
Person person;
person.ParseFromIstream(&input);
cout << "Name: " << person.name() << endl;
cout << "E-mail: " << person.email() << endl;
```

你可以在不打破向后兼容性的情况下向 `message` 中添加新字段. 旧的二进制(old binaries???) 在解析的时候只是简单的忽略新字段. 如果你有一个通信协议,它使用 `protocol buffers` 作为数据格式, 你可以扩展你的 `protocol` 而不需要担心和现有代码的兼容性.

你可以在 [API Reference section](https://developers.google.com/protocol-buffers/docs/reference/overview)中找到使用生成的 `protocol buffer` 的完整参考.在 [Protocol Buffer Encoding](https://developers.google.com/protocol-buffers/docs/encoding) 中找到如何编码 `protocol buffer messages`.


## 为什么不直接使用 XML 呢?
从序列化结构数据的方面来说, `Protocol buffers` 对于 `XML` 是有很多优点的. `Protocol buffers`:
* 更简单
* 小3到10倍
* 快 20到100倍
* 更小的模糊性
* 生成更容易以编程方式使用的数据访问类

例如,假设你想对一个人建模,有一个`name`字段和一个`email`字段. 使用 `XML` ,你需要这么写:
```
<person>
   <name>John Doe</name>
   <email>jdoe@example.com</email>
 </person>
```

对应的`protocol buffer message` 是   (在 `protocol buffer`的 `txt format`下看):
```
# Textual representation of a protocol buffer.
# This is *not* the binary format used on the wire.
person {
  name: "John Doe"
  email: "jdoe@example.com"
}
```

当这个`message`编码成 `protocol buffer binary format` (`txt format` 只是为了方便人类阅读,debug和修改), 它大概只有 28bytes 大小, 只需要大约 100-200 纳秒 解析. `XML`文件需要 至少69bytes来存储,需要 5000-10000 纳秒 解析.

同样,操作 一个 `protocol buffer ` 更加简单:
```c++
cout << "Name: " << person.name() << endl;
cout << "E-mail: " << person.email() << endl;
```

对于XML来说:
```c++
cout << "Name: "
       << person.getElementsByTagName("name")->item(0)->innerText()
       << endl;
cout << "E-mail: "
     << person.getElementsByTagName("email")->item(0)->innerText()
     << endl;
```


## 参考资料
[https://developers.google.com/protocol-buffers/docs/overview](https://developers.google.com/protocol-buffers/docs/overview)
