# 诡异的现象
version: tensorflow1.15, python3.6

`feature_column.shared_embedding_columns` & `feature_column.sequence_numeric_column` 使用 `feature_column.make_parse_example_spec`会报错。错在 `isinstance(col, _FeatureColumn)`上。
