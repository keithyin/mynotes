
# feather 文件

```python
# table 写 feather
import pyarrow.feather as feather
feather.write_feather(df, '/path/to/file.arrow')  # 默认是 lz4压缩。不要压缩的话，使用 compression="uncompressed"


# record batch写
schema = pa.schema([pa.field("v", pa.int32)])

with pa.OSFile("bigfile.arrow", "wb") as sink:
  with pa.ipc.new_file(sink, schema) as writer:
    batch = pa.record_batch([pa.array(range(BATCH_SIZE), type=pa.int32())], schema)
    writer.write(batch)

```

```python
# 读
# Result is pandas.DataFrame。等价于 read_table 然后 convert to pandas
read_df = feather.read_feather('/path/to/file')

# Result is pyarrow.Table
read_arrow = feather.read_table('/path/to/file')


# 读。和read_table等价
with pa.OSFile('bigfile.arrow', 'rb') as source:
   loaded_array = pa.ipc.open_file(source).read_all()


print("LEN:", len(loaded_array))
### LEN: 10000000

print("RSS: {}MB".format(pa.total_allocated_bytes() >> 20))
### RSS: 38MB

# mmap读
with pa.memory_map('bigfile.arrow', 'rb') as source:
   loaded_array = pa.ipc.open_file(source).read_all()


print("LEN:", len(loaded_array))
### LEN: 10000000

print("RSS: {}MB".format(pa.total_allocated_bytes() >> 20))
### RSS: 0MB

```
