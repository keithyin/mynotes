# Storm

* 分布式实时计算系统
* MapReduce 是离线计算系统
* 用来处理信息流，一个 消息过来，就处理一次。
  * 有个缺点，如果数据一次性过来很多，就很难处理，这时候需要一个消息队列技术（kafka）



## 基本概念

* Topologies: 拓扑，也俗称一个任务
* Spouts：拓扑的消息源
* Bolts：拓扑的处理逻辑单元
* Streams：流
* Stream Grouping：流的分组策略
* Tasks：任务处理单元，一个线程可以运行多个 task 实例
* Executors：工作线程
* Wokers：工作进程
* Configuration：topology的配置



**写程序的时候，处理三个部分就可以了：Spouts，Bolts，Topology**

* Spouts 将消息元祖（tuple）传给 bolt
* 数据的流向是个有向无环图



```java
// randomwordspout.java
public class RandomWrodSpout extends BaseRichSpout{
    private SpoutOutputCollector collector;
    
    @override
    public void nextTuple(){
        // 这个函数对不断调用
        collector.emit(new Values("keith"))
    }
    
    @override
    public void open(Map arg0, TopologyContext arg1, SpoutOutputCollector arg2){
        // 初始化方法
        this.collector = arg2;
    }
    
    @override
    public void declareOutputFields(OuputFieldsDeclare arg0){
        // 声明字段的意义
        arg0.declare(new Fields("name"))
    }
}
```



```java
// upperbolt.java
public class UpperBolt extends BaseBasicBolt{
    
    // 业务处理逻辑
    @override
    public void execute(Tuple tuple, BasicOutputCollector collector){
        string goodname = tuple.getString(0)
        collector.emit(new Values("value"))
    }
    
    
    // 定义该 Bolt 发出去的 Tuple 的字段名
    @override
    public void declareOutputFields(OuputFieldsDeclare arg0){    
    }
}
```



```java
// suffixbolt.java
public class SuffixBolt extends BaseBasicBolt{
    
    // 在 bolt 运行过程中只会被调用一次
    @override
    public void prepare(Map stormConf, TopologyContext context){
        
    }
    
    // 业务处理逻辑
    @override
    public void execute(Tuple tuple, BasicOutputCollector collector){
        string goodname = tuple.getString(0)
        collector.emit(new Values("value"))
    }
    
    
    // 定义该 Bolt 发出去的 Tuple 的字段名
    @override
    public void declareOutputFields(OuputFieldsDeclare arg0){   
        
    }
}
```



```java
// topology.java
// topotoly 提交到集群后会永无休止的运行，除非人为退出或异常
public class Topology{
    public static void main(String[] args){
        TopologyBuilder builder = new TopologyBuilder();
        
        //设置 spouts 组件，执行时后有多个 spouts 并行执行的。
        builder.setSpout("randomspot", new RandomSpout(),4);
        
        // 设置大写 Bolts，执行时后有多个 upperbolts 并行执行的。
        builder.setBolt("upperbolt", new UpperBOlt(),4).shuffleGrouping("randomspot");
        
        // 设置写文件的 bolt，执行时后有多个 suffixbolts 并行执行的。
        builder.setBolt("suffixbolt", new SuffixBolt(),4).shuffleGrouping("upperbolt");
        
        StomTopology topology = builder.createTopology();
        
        Config conf = new Config();
        conf.setNumWorkers(4);
        conf.setDebug(true);
        conf.setNumAckers(0);
        StormSubmitter.submitTopology("demotopo", conf, topology);
    }
}
```

