spark上对三种数据的操作
* 对于 数据库表：使用 `spark.sql(sql_str)` 进行查询, 会得到一个 DataFrame 结果。然后使用 DataFrame 的API进行操作
* 对于 DataFrame：有一整套对于 DataFrame 操作的方法
* 对于 RDD：DataFrame.rdd 得到就是 rdd数据表示了，然后可以用rdd操作的一套API

三种数据的互相转换：
* 数据库表 & DataFrame: 
  * 数据库 -> DataFrame
