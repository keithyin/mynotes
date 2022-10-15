# HiveQL

1. 通过查询向表中插入数据

```sql
insert overwrite table o_t partition (country='US', state='OR')
select * from in_t where in_t.cnty='US' and in_t.st='OR'

-- 如果一个源表要插入多个 目标表分区的话, 这样写的好处是，一次扫描 in_t, 可以插入 o_t 的多个分区
from in_t
insert overwrite table o_t partition (country='US', state='OR')
select * from in_t where in_t.cnty='US' and in_t.st='OR'

insert overwrite table o_t partition (country='US', state='CA')
select * from in_t where in_t.cnty='US' and in_t.st='CA'

-- 动态分区插入，写起来比上面更简单. 一次扫描，插入多个分区. 
insert overwrite table o_t partition (country, state)
select *, cnty, st from i_t

-- 混合静态分区 & 动态分区
insert overwrite table o_t partition (country='US', state)
select *, cnty, st from i_t where cnty='US'
```



2. 查询

> where 中不能用别名。可以采用嵌套 select 方式解决

```sql
-- 嵌套 select. 两种写法
select * from 
(select * from t where dt='20220101') e;

from (
	select * from t where dt='20220101'
) e select *;

-- like, rlike (like 简易的正则表达式匹配，rlike：使用了 java 的正则表达式匹配)

-- having 语句，解决需要一个子查询才能对 group by 之后的结果限制输出的情况
select year(ymd), avg(price) from stocks
group by year(ymd)
	having avg(price) > 50.0;
	

-- join. hive会先对 a,b 执行 join 操作，再对 a,b join 得到的中间表 和 c 进行 join 操作。这需要2个 MapReduce 操作
select a.ymd, b.ymd, c.ymd
from stocks a 
join stocks b on a.ymd = b.ymd
join stocks c on a.ymd = c.ymd

-- 上面例子中，由于都用到了 a.ymd 作为其中一个 join 键，在这种情况下，hive 通过一个优化可以在一个 MapReduce 中连接三个表
-- 当对3个及以上表进行join 连接时，如果每个 on 字句都使用了相同的连接键，那么只会产生一个 MapReduce job
-- hive 同时假设最后一个表是最大的表，在对每行记录进行连接操作时，它会尝试将其它表缓存起来，然后扫描最后一个表进行计算。所以用户应该确保连续查询中的表的大小是从左到右递增的 （保证第一个表是最小的就好了。）
-- 即使不是从左到右递增就也行，hive 提供了一个标记表示哪个表是大表  (标记说明 stock 当做大表)
select /*+STREAMTABLE(s)*/ s.ymd, s.symbol, d.dividened
from stocks s join dividends d on s.ymd=d.ymd and s.symbol=d.symbol

-- where 中添加分区过滤器可以增加查询速度

-- outer join 时，分区过滤条件是不能放在 on 语句里的，inner join 时是可以的

-- where 语句在连接操作执行过后才会执行

-- 嵌套 select 语句会按照要求执行下推过程，在数据进行连接操作之前会先进行 分区过滤。所以分区过滤在子查询中做是合理的

-- map-side join：如果所有的表中有一张小表，那么小表可以完全放在内存中，然后在 map 端执行 join 操作。dividends 是个小表，所以可以用此标记。hive0.7之后，该标记被废弃，但是还是有效的。但是 hive0.7之后提供了 set hive.auto.convert.join=true; 来支持该优化。hive 对 (right outer join, full outer join) 不支持该优化
select /*+MAPJOIN(d)*/ *
from stocks s join dividends d on s.ymd=d.ymd

-- order by : 全局排。sort by 局部排序(reducer 内部)。？hive 查询返回的结果，相同 reducer 产出的数据是在一起的吗？

-- distribute by : 控制 map 输出的内容在 reducer 中是如何划分的。在 Streaming 特征 和 UDAF 中可能会用到
select s.ymd, s.symbol, s.price_close
from stocks s
distribute by s.symbol -- 相同的 symbol 会放到同一个 reducer 中
sort by s.symbol, s.ymd  -- 因为一个 reducer 中可能会存在多个 symbol, 所以需要对symbol也排序

-- cluster by : 
cluster by s.symbol 等价于 distribute by s.symbol sort by s.symbol

-- 
```



# UDF UDAF UDTF

* UDF: 输入一行中的 0~N 列数据，然后输出一个值
* UDAF：输入多行的 0~N 列数据，然后输出一个值
* UDTF: 输入一行的 0~N列数据，然后输出 1~M 行，1~K 列数据

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


```java
package com.sankuai.meituan.hive.udf;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;


@Description(
        name = "L2NormalizeUDF",
        value = "_FUNC_(String inp) - Returns l2-normed result"
)

/**
 * select ToJsonUDF(
 *   'a', array(1,2,3), false,
 *   'b', array('1', '2', '3'), false,
 *   'c', '{"first":10, "second": "20", "third": true}', true,
 *   'd', '{"first":10, "second": "20", "third": true}', false,
 *   'e', array('{"four": 10}', '{"five": 5}'), true
 * )
 * 结果。
 * {
 *     "a":[
 *         1,
 *         2,
 *         3
 *     ],
 *     "b":[
 *         "1",
 *         "2",
 *         "3"
 *     ],
 *     "c":{
 *         "third":true,
 *         "first":10,
 *         "second":"20"
 *     },
 *     "d":"{\"first\":10, \"second\": \"20\", \"third\": true}",
 *     "e":[
 *         {
 *             "four":10
 *         },
 *         {
 *             "five":5
 *         }
 *     ]
 * }
 */
public class ToJsonUDF extends GenericUDF {
    private List<List<ObjectInspector>> ois= new ArrayList();

    private List<Boolean> isListFlag = new ArrayList<>();

    private PrimitiveObjectInspector outputOI;
    /**
     * ToJsonUDF(
     *      key1, val1, ori_val_is_json1
     *      key2, val2, ori_val_is_json2
     *      ...
     * )
     * key：JsonKey，必须是String类型
     * val：JsonValue。支持 int,long,float,double,string,boolean. 同时支持与之对应的 list 结构
     * ori_val_is_json1。标识对应的val是不是jsonStr。boolean。
     *
     * @param objectInspectors
     * @return
     * @throws UDFArgumentException
     */
    @Override
    public ObjectInspector initialize(ObjectInspector[] objectInspectors) throws UDFArgumentException {
        if (objectInspectors.length % 3 != 0) {
            throw new RuntimeException("invalid arguments, arguments.length % 3 != 0");
        }
        ois = new ArrayList<>(); // 注意每次调用都要初始化！！这个东西并不是一个对象仅调用一次 initialize
        isListFlag = new ArrayList<>();
        for (int i=0; i < objectInspectors.length; i++) {
            if (i % 3 == 0) {
                ois.add(new ArrayList<>());
            }

            ois.get(ois.size()-1).add(objectInspectors[i]);


            if (i % 3 == 0) {
                if (objectInspectors[i].getCategory() != ObjectInspector.Category.PRIMITIVE ||
                        ((PrimitiveObjectInspector)objectInspectors[i]).getPrimitiveCategory()
                                != PrimitiveObjectInspector.PrimitiveCategory.STRING
                ) {
                    throw new RuntimeException(String.format(
                            "invalid json key type, pos:[%d] is not String", i));
                }
            }

            if (i % 3 == 2) {
                if (objectInspectors[i].getCategory() != ObjectInspector.Category.PRIMITIVE ||
                        ((PrimitiveObjectInspector)objectInspectors[i]).getPrimitiveCategory()
                                != PrimitiveObjectInspector.PrimitiveCategory.BOOLEAN
                ) {
                    throw new RuntimeException(String.format(
                            "invalid json key type, pos:[%d] is not boolean", i));
                }
            }

            if (i % 3 == 1) {
                if (objectInspectors[i].getCategory() == ObjectInspector.Category.PRIMITIVE) {
                    isListFlag.add(false);
                } else if (objectInspectors[i].getCategory() == ObjectInspector.Category.LIST) {

                    isListFlag.add(true);
                } else {
                    throw new RuntimeException("");
                }
            }

        }

        outputOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        return outputOI;
    }

    @Override
    public Object evaluate(DeferredObject[] deferredObjects) throws HiveException {
        List<List<DeferredObject>> values = new ArrayList<>();
        for (int i = 0; i < deferredObjects.length; i++) {
            if (i % 3 == 0) {
                values.add(new ArrayList<>());
            }
            values.get(values.size()-1).add(deferredObjects[i]);
        }

        if (ois.size() != values.size()) {
            throw new RuntimeException(String.format("OI.size(%d) != values.size(%d)", ois.size(), values.size()));
        }

        JSONObject jo = new JSONObject();
        try {
            for (int i = 0; i < ois.size(); i++) {
                PrimitiveObjectInspector keyOI = (PrimitiveObjectInspector) ois.get(i).get(0);
                PrimitiveObjectInspector isJsonStrOI = (PrimitiveObjectInspector) ois.get(i).get(2);
                ObjectInspector valueOI = ois.get(i).get(1);
		// DeferredObject 需要.get() 才能将 Object 取出来。OI 可以对其进行 getPrimitiveJavaObject 操作！
                String key = keyOI.getPrimitiveJavaObject(values.get(i).get(0).get()).toString();
                boolean isJsonStr = (boolean) isJsonStrOI.getPrimitiveJavaObject(values.get(i).get(2).get());

                Object deferredVal = values.get(i).get(1).get();
                if (deferredVal == null) continue;

                if (isJsonStr) {
                    // jsonStr or list of jsonStr
                    if (valueOI.getCategory() == ObjectInspector.Category.PRIMITIVE) {
                        PrimitiveObjectInspector valuePOI = (PrimitiveObjectInspector) valueOI;
                        String value = valuePOI.getPrimitiveJavaObject(deferredVal).toString();
                        jo.put(key, new JSONObject(value));

                    } else if (valueOI.getCategory() == ObjectInspector.Category.LIST) {
                        ListObjectInspector valueLOI = (ListObjectInspector) valueOI;
                        PrimitiveObjectInspector listElementValuePOI = (PrimitiveObjectInspector) valueLOI.getListElementObjectInspector();
                        JSONArray jsonArray = new JSONArray();
                        jo.put(key, jsonArray);
                        for (int eleIdx = 0; eleIdx < valueLOI.getListLength(deferredVal); eleIdx++) {
                            Object listElementVal = valueLOI.getListElement(deferredVal, eleIdx);
                            JSONObject inner = new JSONObject(listElementValuePOI.getPrimitiveJavaObject(listElementVal).toString());
                            jsonArray.put(eleIdx, inner);
                        }
                    }

                } else { // 如果不是 jsonStr，那么支持两种（主类型，List类型）. 当前仅是支持一层 List。还不支持 List of json str

                    if (valueOI.getCategory() == ObjectInspector.Category.LIST) {
                        ListObjectInspector valueLOI = (ListObjectInspector) valueOI;
                        PrimitiveObjectInspector listElementValuePOI = (PrimitiveObjectInspector) valueLOI.getListElementObjectInspector();
                        JSONArray jsonArray = new JSONArray();
                        jo.put(key, jsonArray);
                        for (int eleIdx = 0; eleIdx < valueLOI.getListLength(deferredVal); eleIdx++) {
                            Object listElementVal = valueLOI.getListElement(deferredVal, eleIdx);
                            switch (listElementValuePOI.getPrimitiveCategory()) {
                                case INT:
                                    jsonArray.put(eleIdx, (int) listElementVal);
                                    break;
                                case LONG:
                                    jsonArray.put(eleIdx, (long) listElementVal);
                                    break;
                                case FLOAT:
                                    jsonArray.put(eleIdx, (float) listElementVal);
                                    break;
                                case DOUBLE:
                                    jsonArray.put(eleIdx, (double) listElementVal);
                                    break;
                                case STRING:
                                    jsonArray.put(eleIdx, listElementVal.toString());
                                    break;

                                case BOOLEAN:
                                    jsonArray.put(eleIdx, (boolean) listElementVal);
                                    break;
                                default:
                                    throw new RuntimeException(String.format("not supported type %s", listElementValuePOI.getPrimitiveCategory()));

                            }

                        }


                    } else {
                        PrimitiveObjectInspector valuePOI = (PrimitiveObjectInspector) valueOI;

                        switch (valuePOI.getPrimitiveCategory()) {
                            case INT:
                                jo.put(key, (int) valuePOI.getPrimitiveJavaObject(deferredVal));
                                break;
                            case LONG:
                                jo.put(key, (long) valuePOI.getPrimitiveJavaObject(deferredVal));
                                break;
                            case FLOAT:
                                jo.put(key, (float) valuePOI.getPrimitiveJavaObject(deferredVal));
                                break;
                            case DOUBLE:
                                jo.put(key, (double) valuePOI.getPrimitiveJavaObject(deferredVal));
                                break;
                            case STRING:
                                jo.put(key, valuePOI.getPrimitiveJavaObject(deferredVal).toString());
                                break;

                            case BOOLEAN:
                                jo.put(key, (boolean) valuePOI.getPrimitiveJavaObject(deferredVal));
                                break;
                            default:
                                throw new RuntimeException(String.format("not supported type %s", valuePOI.getPrimitiveCategory()));

                        }

                    }

                }

            }
        }catch (JSONException e) {
            throw new RuntimeException(e.toString());
        }

        return jo.toString();
    }

    @Override
    public String getDisplayString(String[] strings) {
        return null;
    }
}
	

```


## UDAF



## UDTF

https://riptutorial.com/hive/example/22316/udtf-example-and-usage



## 参考资料

https://blog.dataiku.com/2013/05/01/a-complete-guide-to-writing-hive-udf

https://www.hadoopdoc.com/hive/hive-udf-intro
