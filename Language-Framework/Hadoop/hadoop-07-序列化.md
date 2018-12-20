# hadoop 中的序列化机制

* `hadoop` 中的数据大量涉及了网络数据传输
* 自定义类型的话 需要 实现 `hadoop` 自定义的接口
* `hadoop` 序列化不保存类的继承机构

```java
public class FlowBean implements Writable, Comparable{
    private String name = null;
    private long up_flow=0;
    // 结构数据 --> 序列化
    @override
    public void write(DataOutput out) throw IOException{
    	out.writeUTF(name);
        out.writeLong(up_flow);
    }
    
    // 序列化 --> 结构数据
    @override
    public void readFields(DataInput in) throw IOException{
        // 从数据流中读对象字段时，顺序和序列化时候的要一致
        name = in.readUTF();
        up_flow = in.readLong();
    }
    
    // 写文件的时候会调用这个函数！！！！
    @override
    public String toString(){
        return "\t"+ upflow + "\t"+ "bits";
    }
    
}
```



# 自定义排序实现

* 默认是按照 `key` 的字典序排序的
* 排序的时候，会使用 `key` 类型的 `compareTo` 方法
* 可以通过自定义`Key` 类型的方式，`key` 类型需要实现 `Comparable`（Java原生接口） 



# 自定义分组实现

* `map` 输出的 `key-value` 是通过 `key` 分组然后给 `reduce` 的，是 `mr` 框架自带的
  * 默认是通过 `HashPartitioner` 来区分的。

```java
// hadoop
public class HashPartitioner<K,V> extends Partitioner<K,V>{
    public int getPartition(K key, V value, int numReduceTasks){
        return (key.hashCode()&Integer.MAX_VLAUE)%numReduceTasks;
    }
}
```



**如何将不同的组的统计结果输出到不同文件**

* 自定义 `Partitioner`
* 自定义 `Reducer` 的并发任务数
  * 默认只有一个
  * 在 `job` 中控制，`job.setNumReduceTasks(6);` 和分组的个数保持一致

```java
// AreaPartitioner.java
// 在 Runner 中配置此 Partitioner上，job.setPartitionerClass(AreaPartitioner.class)
public class AreaPartitioner<KEY, VALUE> extends Partitioner<KEY,VALUE>{
    @override
    public int getPartition(KEY key, VALUE value, int numPartitions){
        
    }
}
```



