spark上有对三种数据类型的操作

* 对于 数据库表：使用 `spark.sql(sql_str)` 进行查询, 会得到一个 DataFrame 结果。然后使用 DataFrame 的API进行操作
* 对于 DataFrame：有一整套对于 DataFrame 操作的方法
* 对于 RDD：DataFrame.rdd 得到就是 rdd数据表示了，然后可以用rdd操作的一套API

三种数据的互相转换：
* 数据库表 & DataFrame: 
  * 数据库表 -> DataFrame: `spark.sql(sql_str)` 返回的就是 `DataFrame`
  * DataFrame -> 数据库表： `df.createOrReplaceTempView("view_name")`. 执行该操作后就可以对 `view_name` 进行 sql 操作了。
* DataFrame & RDD
  * DataFrame -> RDD: `df.rdd` 得到的就是RDD
  * RDD -> DataFrame: `df = spark.createDataFrame(res_rdd, schema)` 得到的就是 `DataFrame` 了
* 数据库表 & RDD：之间好像没有直接转的，就把 `DataFrame` 当作桥梁用吧。

动作操作：`count(), show(), take()`
