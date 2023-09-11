

# 读写 for IPC
在IPC传递数据时，主要是 序列化与反序列化。 这部分主要是介绍用于 IPC 的数据

## 流式读写

```python
import pyarrow as pa

data = [
    pa.array([1, 2, 3, 4]),
    pa.array(['foo', 'bar', 'baz', None]),
    pa.array([True, None, False, True])
]

batch = pa.record_batch(data, names=['f0', 'f1', 'f2'])

batch.num_rows

batch.num_columns

# new_stream 创建一个 RecordBatchStreamWriter 往 stream 中写数据
# which can write to a writeable NativeFile object or a writeable Python object
sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, batch.schema) as writer:
   for i in range(5):
      writer.write_batch(batch)

buf = sink.getvalue()
buf.size

# 基于 buffer 开启一个流，用来读取
with pa.ipc.open_stream(buf) as reader:
    schema = reader.schema
    batches = [b for b in reader]
schema
len(batches)


# 随机写
sink = pa.BufferOutputStream()

with pa.ipc.new_file(sink, batch.schema) as writer:
   for i in range(5):
      writer.write_batch(batch)
buf = sink.getvalue()
buf.size

# 随机读
with pa.ipc.open_file(buf) as reader:
  num_record_batches = reader.num_record_batches

  b = reader.get_batch(3)

```

## 文件读写
高效读写 arrow 数据

```python

# 写 arrow 数据
BATCH_SIZE = 10000

NUM_BATCHES = 1000

schema = pa.schema([pa.field('nums', pa.int32())])

with pa.OSFile('bigfile.arrow', 'wb') as sink:
  with pa.ipc.new_file(sink, schema) as writer:
    for row in range(NUM_BATCHES):
      batch = pa.record_batch([pa.array(range(BATCH_SIZE), type=pa.int32())], schema)
      writer.write(batch)

# 读 arrow 数据， 读取到 内存中
with pa.OSFile('bigfile.arrow', 'rb') as source:
  loaded_array = pa.ipc.open_file(source).read_all()
print("LEN:", len(loaded_array))
## LEN: 10000000

print("RSS: {}MB".format(pa.total_allocated_bytes() >> 20))
## RSS: 38MB

# 读 arrow 数据，mmap到内存中, 读取的时候才会通过page fault将页加载到内存中
with pa.memory_map('bigfile.arrow', 'rb') as source:
  loaded_array = pa.ipc.open_file(source).read_all()

print("LEN:", len(loaded_array))
## LEN: 10000000

print("RSS: {}MB".format(pa.total_allocated_bytes() >> 20))
## RSS: 0MB
```

## feather 文件格式
> 将ipc数据直接dump到文件中

```python

# 写文件
import pyarrow.feather as feather
feather.write_feather(df_or_pa_table, '/path/to/file')

# 读文件
# Result is pandas.DataFrame
read_df = feather.read_feather('/path/to/file')

# Result is pyarrow.Table
read_arrow = feather.read_table('/path/to/file')


with open('/path/to/file', 'wb') as f:
    feather.write_feather(df, f)

with open('/path/to/file', 'rb') as f:
    read_df = feather.read_feather(f) # read_table 也可以

```

```python
# 压缩

# Uses LZ4 by default
feather.write_feather(df, file_path)

# Use LZ4 explicitly
feather.write_feather(df, file_path, compression='lz4')

# Use ZSTD
feather.write_feather(df, file_path, compression='zstd')

# Do not compress
feather.write_feather(df, file_path, compression='uncompressed')


# 读，就这一个就可以了。
read_feather(). 

```


