# TensorFlow Data Validation

> 用来验证训练数据的, 主要是为了验证结构化数据

```python
!pip install -q tensorflow_data_validation
import tensorflow as tf
import tensorflow_data_validation as tfdv

tf.logging.set_verbosity(tf.logging.ERROR)
print('TFDV version: {}'.format(tfdv.version.__version__))

# 从 csv 文件中统计信息
train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)

# 可视化 数据的统计信息 (这个太帅了)
tfdv.visualize_statistics(train_stats)
```

**概念**

* `schema` : 数据集的 `schema`, 比如: 特征值的类型, 特征是数值型还是类别型, `or the frequency of its presence in the data?????`
  * 对于类别型特征来说, `schema` 指定了其 `domin` (可取值集合)
* 总之: 一个数据集的 `schema` 包含了
  * 特征名, 特征的类型(INT, FLOAT, STRING), 类别特征的`domin` 等等

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```



### 检查验证集数据

**可能存在问题**

* 数据集的 `schema` 不一致
  * 类别特征 是不是有 diff?
* 训练集, 验证集 数据分布不一致
  * 均值不一样? 方差不一样? 最大最小值不一样? 分位数不一样?



```python
# 检查 schema 之间的差异
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=train_schema)
tfdv.display_anomalies(anomalies)
```



```python
# 检查数据分布, 看看是不是不一样 Compute stats for evaluation data
eval_stats = tfdv.generate_statistics_from_csv(data_location=EVAL_DATA)

# Compare evaluation data with training data
tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')
```



```python
# 修改 scheme
# Relax the minimum fraction of values that must come from the domain for feature company.
# 所以 这个操作是干啥的???????
company = tfdv.get_feature(schema, 'company')
company.distribution_constraints.min_domain_mass = 0.9

# Add new value to the domain of feature payment_type.
payment_type_domain = tfdv.get_domain(schema, 'payment_type')
payment_type_domain.value.append('Prcard')

# Validate eval stats after updating the schema 
updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
tfdv.display_anomalies(updated_anomalies)
```



**如果训练集 和 测试集 本身就存在 特征不一致怎么办?  比如测试集没有label这个特征**

* `schema` 缺了特征 怎么办 (不是缺乏特征值哦)
* 如果训练集比测试集多了个特征, 这时候怎么能通过 `tfdv.validate_statistics` 呢
  * `Environment` 来解决这个问题: `features in schema` 可以和 `environments` 关联起来, 

```python
# 保证类型一致
options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA, stats_options=options)
serving_anomalies = tfdv.validate_statistics(serving_stats, schema)

tfdv.display_anomalies(serving_anomalies)
```

```python
# All features are by default in both TRAINING and SERVING environments.
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# Specify that 'tips' feature is not in SERVING environment.
tfdv.get_feature(schema, 'tips').not_in_environment.append('SERVING')

serving_anomalies_with_env = tfdv.validate_statistics(
    serving_stats, schema, environment='SERVING')

tfdv.display_anomalies(serving_anomalies_with_env)
```

