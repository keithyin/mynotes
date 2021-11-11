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
  * RDD -> DataFrame: `df = spark.createDataFrame(res_rdd, schema)` 得到的就是 `DataFrame` 了； 或者 `rdd.toDF()` 可以得到 `DataFrame`
* 数据库表 & RDD：之间好像没有直接转的，就把 `DataFrame` 当作桥梁用吧。

动作操作：`count(), show(), take()`


# DataFrame
DataFrame的schema定义了DataFrame的列名和类型
```python
import pyspark.sql.types as T

# StructType 可以套 StructType。估计不行 🙅
schema = T.StructType([
    T.StructField("user_id", T.LongType(), True),
    T.StructField("coupon_id", T.LongType(), True),
    T.StructField("info", T.ArrayType(T.StringType(), True), True)])
```

DataFrame的列：表示一个简单类型(整数，字符串 ..)或者一个复杂类型 (Array，Map)

* `StructType`: 一个 `StructField` array
* `StructField`: 表示 `DataFrame` 某列的字段描述
* `**Type`: 表示数据类型

对DataFrame的操作主要是类似 sql 语句中的操作，不过是需要通过 `selectExpr` 方式来调用。
* 字段的操作
* 条件语句： `df.where("col_name < expr")`
* 聚合语句: `df.groupBy("col_name")` 聚合语句后面可以接着一堆聚合操作。`df.groupBy('col_name1').sum('col_name2')....`

# RDD
row 类型 RDD？ 难道还有其它类型
对于row类型，可以 `row[0]`, 可以 `row.col_name`，可以 `row['colname']`
Row can be used to create a row object by using named arguments. It is not allowed to omit a named argument to represent that the value is None or missing. This should be explicitly set to None in this case.

```python
from pyspark.sql import Row
row = Row(name="Alice", age=11)
print(row['name'], row['age'])
print(row.name, row.age)
print('name' in row)
print('wrong_key' in row)

Person = Row("name", "age")
print('name' in Person)
print('wrong_key' in Person)
Person("Alice", 11)
```



# 窗口函数

```sql
aggr_func() over ([partition by ] [order by])
```



```sql
select 
 user_id,
 price,
 sum(price) over (partition by user_id) as user_tot_price
from some_table
```

* `over (partition by )` 用来表示表示窗口。` partition by user_id` 既表示了窗口切分规则。也表示当前记录应该属于哪个窗口
* `sum()` 用于该窗口的聚合函数



可用的聚合函数

```sql
row_number()  over(partition by x order by y) 
rank()  over(partition by x order by y)         --分相同的排名一样，但是后面的名次会跳跃
dense_rank()over(partition by x order by y)     --分相同的排名一样，且后面名次不跳跃
first_value() over(partition by x order by y)   --第一次出现的值赋值给 本窗口内的所有记录
last_value() over(partition by x order by y)    
```

