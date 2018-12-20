# map reduce

* map： 局部处理工作，多个节点同时运行 map
* reduce：汇总工作，一个 reduce 或者多个 reduce 由真实业务决定
* map 和 reduce 的输入输出都是以 key-value 的形式封装的
* map 的输入类型由框架控制，默认情况下，框架传递给 Map 的数据中，key是要处理的的文本中一行的起始偏移量，这一行的内容作为 value

```shell
# data.txt
hello world
hello keith
```



**如何写代码**

* 自定义一个 `CustomMapper` 类，继承 `Mapper`
* 自定义一个 `CustomReducer` 类，继承 `Reducer`
* 定义一个 `Runner`，该配置的配置
* 然后将项目打成 `jar` 包

```java
// word count 的 map reduce 程序
// WCMapper.java
// Mapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT>
// 使用 hadoop 自己封装的 Long，String，序列化的时候更精简
public class WCMapper extends Mapper<LongWritable, Text, Text, LongWritable>{
    
    // map reduce 框架每读一行数据就调用一次该方法
    @override
    protected void map(LongWritable key, Text value, Context context) throws Exception{
    	// 编写具体业务逻辑
        // key：这一行数据的起始偏移量
        // value：这一行的文本内容
        // context: 输出的一些工具
        
        // 转换成 String 类型，然后 java 的操作
        String line = value.toString();
        String[] words = StringUtils.split(line, " ");
        for (String word : words){
            context.write(new Text(word), new LongWritable(1));
        }
    }
}
// 每个 key 的结果都出来后，才会调用 reduce，所以应该就是 map 执行完毕才会 reduce
//WCReduccer.java

public vlass WCReducer extends Reducer<Text, LongWritable, Text, LongWritable>{
    // 框架在 map 处理完成之后，将所有 kv 对缓存起来，进行分组，然后传递一个组，调用一次
    // 一组：key：values
    // <hello: {1,1,1,1,1}>
    @override
    protected void reduce(Text key, Iterable<LongWritable> arg1, Context arg2){
        long count = 0;
        // 遍历 value list 进行累加求和
        for (LongWritable v: arg1){
            count += v.get();
        }
        // 输出一个单词的统计结果
        arg2.write(key, new LongWritable(count));
    }
}

// WCRunner.java
// 用来描述一个特定的作业，
// 比如改作业使用哪个类作为 逻辑处理的 map，哪个作为 reducer
// 还可以指定该作业要处理的数据所在的路径
// 还可以指定该作业输出的结果放在哪个路径
public class WCRunner{
    public static void main(String[] args){
        Configuration conf = new Configuration()
        Job job = Job.get(conf);
        // 设置整个 job 所用的那些类所用的 jar 包
        job.setJarByClass(WCRunner.class);
        // 本Job 使用的 map 和 reduce 的类
        job.setMapperClass(WCMapper.class);
        job.setReducerClass(WCReducer.class);
        
        //对 map 和 reduce 的输出都起作用
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        
        // 可以单独指定 map 的输出类型
        
        //指定原始数据放在哪里
        FileInputFormat.setInputPaths(job, new Path("/wc/"));
        
        // 指定输出数据存放路径
        FileOuputFormat.setOuputPath(job, new Path("/wc/output/"));
        
        // 将 job 提交给集群运行
        job.waitForCompletion(true);
    }
}

/**
项目打成 jar 包
传到 服务器上面
hadoop jar wc.jar packagepath.WCRunner // 这样就分发执行了。
*/
```

* Map:
  * 框架读文本文件，每读一行，就会发给 Map，让他运行一次
  * 



```java
// 这么实现 runner 也 OK, job 提交的规范写法
public class Runner extends Configured implements Tool{
    @override
    public int run(String[] args) throw Exception{
        /**
        .....
        */
      return job.waitForCompletion(true)?0 else 1; 
    }
    public static void main(String[] args){
        int res = ToolRunner.run(new Configuration(), new Runner(), args);
    	System.exit(res);
    }
}
```



**MapReduce 的本地模式**

* 直接执行 `Runner` 中的 `main` 方法就可以，不用打成 `jar` 包分发。
  * 只要没有 明确的 配置文件，就可以在本地运行
* 数据在本地运行，数据可以在任何文件系统里。



**Yarn 集群**

* 主要是做资源调度的，
* resource manager：用于任务管理
* node manager：分配运行资源容器

