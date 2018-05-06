# opencv 工具类 Directory

`opencv2` 提供了一个 `Directory` 工具类来帮助读取文件夹中的文件名，这还是挺方便的，因为 目前常用的 `c++` 版本并没有提供简单的文件名读取接口。

`Directory`类包含三个静态函数，分别是：

* `static std::vector< std::string > 	GetListFiles (const std::string &path, const std::string &exten="*", bool addPath=true)`
* `static std::vector< std::string > 	GetListFilesR (const std::string &path, const std::string &exten="*", bool addPath=true)`
* `static std::vector< std::string > 	GetListFolders (const std::string &path, const std::string &exten="*", bool addPath=true)`

## static std::vector< std::string > 	GetListFiles (const std::string &path, const std::string &exten="*", bool addPath=true)

这个函数只获取指定目录下的**文件**, 不理会**目录**。

* path：`string`, 用于指定根目录
* exten: `string`,这个是个正则表达式，匹配的返回，否则不返回。
* addPath: `bool`,如果为`true`，返回的文件名会带`path`，如果为`false`，返回的仅是文件名。

## static std::vector< std::string > 	GetListFilesR (const std::string &path, const std::string &exten="*", bool addPath=true)

这个API比上面多了个`R`，这个`R`代表的`recursive`的意思，就是碰到文件夹还是往里钻。

## static std::vector< std::string > 	GetListFolders (const std::string &path, const std::string &exten="*", bool addPath=true)

这个`API` 就是获取指定文件夹下的 文件夹了。
