# python xml 解析

> XML 指可扩展标记语言（e**X**tensible **M**arkup **L**anguage）。 
>
> XML 被设计用来传输和存储数据。
>
> XML是一套定义语义标记的规则，这些标记将文档分成许多部件并对这些部件加以标识。
>
> 它也是元标记语言，即定义了用于定义其他与特定领域有关的、语义的、结构化的标记语言的句法语言。



## ElementTree 解析 xml 文件

```python
# 引入包
import xml.etree.ElementTree as ET
```

要解析的文件为：`movie.xml`

```xml
<collection shelf="New Arrivals">
<movie title="Enemy Behind">
   <type>War, Thriller</type>
   <format>DVD</format>
   <year>2003</year>
   <rating>PG</rating>
   <stars>10</stars>
   <description>Talk about a US-Japan war</description>
</movie>
<movie title="Transformers">
   <type>Anime, Science Fiction</type>
   <format>DVD</format>
   <year>1989</year>
   <rating>R</rating>
   <stars>8</stars>
   <description>A schientific fiction</description>
</movie>
<movie title="Trigun">
   <type>Anime, Action</type>
   <format>DVD</format>
   <episodes>4</episodes>
   <rating>PG</rating>
   <stars>10</stars>
   <description>Vash the Stampede!</description>
</movie>
<movie title="Ishtar">
   <type>Comedy</type>
   <format>VHS</format>
   <rating>PG</rating>
   <stars>2</stars>
   <description>Viewable boredom</description>
</movie>
</collection>
```





```python
# findall, 
# ElementTree.findall(), 查找root 下 所有名字为 path 的节点。
# Element.findall(path) 查找当前元素下一级 的所有名字为 path 的节点 
tree = ET.parse('movie.xml')
objs = tree.findall('movie') 

# 查找第一个满足的 元素
obj = tree.find('movie')

# 如果 obj是个文本节点，则输出里面的内容，否则输出空
print(obj.text)
```

