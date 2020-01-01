# 输入-模型-Trainer

> 这三个部分一起看才好掌握allennlp的脉络, trianer将输入和模型联通起来



### Trainer

```python
class Trainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,  
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None) -> None:

```

* 先介绍`Trainer` 提供了什么功能
  * `tensorboard` 记录训练过程
  * 保存模型
  * `learning_rate_scheduler`
  * `metric`
* 再介绍参数
  * `serialization_dir` : 放序列化模型和`summary`, 如果为 `None`, 这两个大功能都不会生效
* 使用时需要注意的
  * `Model`
    * 继承`allennlp.models.Model`
    * `forward` 的输入是 `Dict[field_name, Dict[indexer_name, tensor]]`
    * `return` 的也需要是个 `Dict` , 对于训练来说, 必须包含 `output["loss"]`  



# 使用Metric

* 重写 `Model.get_metrics(reset)` 方法
  * 一个 `epoch` 结束的时候, `trainer` 会调用 `model.get_metrics(reset=True)`
* 使用方法
  * `model.__init__ ` 时候创建`Metric` 对象
  * `model.forward()` 时候计算一个 batch 的 `metric`
  * `model.get_metrics` 返回.
* `allennlp.training.metrics`

```python
class Demo(model):
  def __init__(self, ...):
    self.accuracy = CategoricalAccuracy()
  def forward(self, ...):
    self.accuracy(pred, ground_truth)
  def get_metrics(self, reset):
    return {"accuracy": self.accuracy.get_metric(reset)}
```

