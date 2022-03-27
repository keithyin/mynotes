# 自定义方法

* UDF: 输入一行中的 0~N 列数据，然后输出一个值
* UDAF：输入多行的 0~N 列数据，然后输出一个值
* 

## UDF

> A UDF processes zero or several columns of one row and outputs one value

实现 UDF 的两种方式

1. 继承 `org.apache.hadoop.hive.ql.exec.UDF` 类，实现 `evaluate()` 方法

   1. `evaluate` 可以重载

   2. | **Hive column type** | **UDF types**                                                |
      | :------------------- | :----------------------------------------------------------- |
      | string               | java.lang.String, org.apache.hadoop.io.Text                  |
      | int                  | int, java.lang.Integer, org.apache.hadoop.io.IntWritable     |
      | boolean              | bool, java.lang.Boolean, org.apache.hadoop.io.BooleanWritable |
      | array<type>          | java.util.List<Java type>                                    |
      | map<ktype, vtype>    | java.util.Map<Java type for K, Java type for V>              |
      | struct               | Don't use Simple UDF, use GenericUDF                         |
      | binary               |                                                              |

2. 继承 `org.apache.hadoop.hive.ql.udf.generic.GenericUDF`

   1. 重点是理解 `ObjectInspector`: Object Inspectors belong to one of the following categories:
      1. Primitive, for primitive types (all numerical types, string, boolean, …)
      2. List, for Hive arrays
      3. Map, for Hive maps
      4. Struct, for Hive structs
   2. 重写 `initialize`: 接收的是输入列的 `OI` ，返回输出列的 `OI`
   3. 



## UDAF



## UDTF

https://riptutorial.com/hive/example/22316/udtf-example-and-usage



# 参考资料

https://blog.dataiku.com/2013/05/01/a-complete-guide-to-writing-hive-udf
