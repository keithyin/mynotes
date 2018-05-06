# dataset

`Dataset`:  用于

```python
class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column or field, together
            with the corresponding Field object. Two fields with the same Field object
            will have a shared vocabulary.
    """
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.
        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        # 保存了 examples 和 fields， 一个 Example 就是一个样本
        # Iterator 使用 __getitem__ 获取数据
        self.examples = examples
        self.fields = dict(fields)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        """工具函数，用来一次创建多个 Dataset。因为在数据集方面，一般分为 训练集，验证集，测试集，
        有这么一个简单的函数来一次性创建三个 Dataset 实例，那是十分方便的。

        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.

        Returns:
            split_datasets (tuple(Dataset)): Datasets for train, validation, and
                test splits in that order, if provided.
        """
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
```





## data.Example

**定义了单个训练/测试样本**

* 用 `Field` 对象对单个样本进行了预处理

```python
class Example(object):
    """Defines a single training or test example.
	将样本的每列(field) 保存成属性，从 cls.fromlist 代码中可以看出端倪
    """

    @classmethod
    def fromJSON(cls, data, fields):
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def fromTSV(cls, data, fields):
        return cls.fromlist(data.split('\t'), fields)

    @classmethod
    def fromCSV(cls, data, fields):
        data = data.rstrip("\n")
        # If Python 2, encode to utf-8 since CSV doesn't take unicode input
        if six.PY2:
            data = data.encode('utf-8')
        # Use Python CSV module to parse the CSV line
        parsed_csv_lines = csv.reader([data])

        # If Python 2, decode back to unicode (the original input format).
        if six.PY2:
            for line in parsed_csv_lines:
                parsed_csv_line = [six.text_type(col, 'utf-8') for col in line]
                break
        else:
            parsed_csv_line = list(parsed_csv_lines)[0]
        return cls.fromlist(parsed_csv_line, fields)

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # 将 text 或 label 通过 field 的设定进行预处理。
                setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [' '.join(t.leaves()), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([' '.join(tree.leaves()), tree.label()], fields)
```



## 自定义 Dataset

要做的事情就是：

准备好 `Example list` 就可以了。。。。

