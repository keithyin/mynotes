# slim dataset

> slim dataset 是对输入流水线的一个高级封装，但是感觉还不如直接用原生的输入流水线好用



## slim.dataset

> 可以看出，dataset 实际上就是一个 dict
>
> data_sources，保存的是 数据文件名，（tfrecords 文件或是其它文件）
>
> reader，就是reader对象，用来读取文件中数据的
>
> decoder，用于解析数据的
>
> num_samples，样本数量

```python
class Dataset(object):
  """Represents a Dataset specification."""

  def __init__(self, data_sources, reader, decoder, num_samples,
               items_to_descriptions, **kwargs):
    """Initializes the dataset.

    Args:
      data_sources: A list of files that make up the dataset.
      reader: The reader class, a subclass of BaseReader such as TextLineReader
        or TFRecordReader.
      decoder: An instance of a data_decoder.
      num_samples: The number of samples in the dataset.
      items_to_descriptions: A map from the items that the dataset provides to
        the descriptions of those items.
      **kwargs: Any remaining dataset-specific fields.
    """
    kwargs['data_sources'] = data_sources
    kwargs['reader'] = reader
    kwargs['decoder'] = decoder
    kwargs['num_samples'] = num_samples
    kwargs['items_to_descriptions'] = items_to_descriptions
    self.__dict__.update(kwargs)
```



## DatasetDataProvider



## ParallelReader

