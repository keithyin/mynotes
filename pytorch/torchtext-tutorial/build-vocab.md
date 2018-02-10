# build vocab

**build vocab 的时候到底干了啥？**



```python
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
```



`build_vocab` 源码

```python
# Field.build_vocab
def build_vocab(self, *args, **kwargs):
    """Construct the Vocab object for this field from one or more datasets.

    Arguments:
        Positional arguments: Dataset objects or other iterable data
            sources from which to construct the Vocab object that
            represents the set of possible values for this field. If
            a Dataset object is provided, all columns corresponding
            to this field are used; individual columns can also be
            provided directly.
        Remaining keyword arguments: Passed to the constructor of Vocab.
    """
    # 用来计数的 dict，
    counter = Counter()
    sources = []
    for arg in args:
        if isinstance(arg, Dataset):
            # arg.fields.items : {'text':field_obj, 'label':field_obj}
            sources += [getattr(arg, name) for name, field in
                        arg.fields.items() if field is self]
        else:
            sources.append(arg)
    for data in sources:
        for x in data:
            if not self.sequential:
                x = [x]
            counter.update(x)
    # counter : key: word token， value: word token 出现的 频数
    specials = list(OrderedDict.fromkeys(
        tok for tok in [self.unk_token, self.pad_token, self.init_token,
                        self.eos_token]
        if tok is not None))
    self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
```



## Vocab

```python
def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
             vectors=None):
    """Create a Vocab object from a collections.Counter.

    Arguments:
        counter: collections.Counter object holding the frequencies of
            each value found in the data.
        max_size: The maximum size of the vocabulary, or None for no
            maximum. Default: None.
        min_freq: The minimum frequency needed to include a token in the
            vocabulary. Values less than 1 will be set to 1. Default: 1.
        specials: The list of special tokens (e.g., padding or eos) that
            will be prepended to the vocabulary in addition to an <unk>
            token. Default: ['<pad>']
        vectors: One of either the available pretrained vectors
            or custom pretrained vectors (see Vocab.load_vectors);
            or a list of aforementioned vectors
    """
    self.freqs = counter
    # counter 中包含了语料库中的 所有 token。
    counter = counter.copy()
    min_freq = max(min_freq, 1)
    counter.update(specials)
	# 顺序 dict
    self.stoi = defaultdict(_default_unk_index)
    self.stoi.update({tok: i for i, tok in enumerate(specials)})
    # 用这个可以将 index 解码成 string
    self.itos = list(specials)

    counter.subtract({tok: counter[tok] for tok in specials})
    max_size = None if max_size is None else max_size + len(self.itos)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(self.itos) == max_size:
            break
        self.itos.append(word)
        self.stoi[word] = len(self.itos) - 1

    self.vectors = None
    if vectors is not None:
        self.load_vectors(vectors)
```



```python
def load_vectors(self, vectors):
    """
    Arguments:
        vectors: one of or a list containing instantiations of the
            GloVe, CharNGram, or Vectors classes. Alternatively, one
            of or a list of available pretrained vectors:
                charngram.100d
                fasttext.en.300d
                fasttext.simple.300d
                glove.42B.300d
                glove.840B.300d
                glove.twitter.27B.25d
                glove.twitter.27B.50d
                glove.twitter.27B.100d
                glove.twitter.27B.200d
                glove.6B.50d
                glove.6B.100d
                glove.6B.200d
                glove.6B.300d
    """
    if not isinstance(vectors, list):
        vectors = [vectors]
    for idx, vector in enumerate(vectors):
        if six.PY2 and isinstance(vector, str):
            vector = six.text_type(vector)
        if isinstance(vector, six.string_types):
            # Convert the string pretrained vector identifier
            # to a Vectors object
            if vector not in pretrained_aliases:
                raise ValueError(
                    "Got string input vector {}, but allowed pretrained "
                    "vectors are {}".format(
                        vector, list(pretrained_aliases.keys())))
            vectors[idx] = pretrained_aliases[vector]()
        elif not isinstance(vector, Vectors):
            raise ValueError(
                "Got input vectors of type {}, expected str or "
                "Vectors object".format(type(vector)))

    tot_dim = sum(v.dim for v in vectors)
    # len(self) 表示当前 corpus 加上一些特殊 token 的长度
    # self.vectors 保存的是 当前 corpus 和 一些 token 的 word embedding
    self.vectors = torch.Tensor(len(self), tot_dim)
    for i, token in enumerate(self.itos):
        start_dim = 0
        for v in vectors:
            end_dim = start_dim + v.dim
            # 用 token 去 vectors 中去索引。
            self.vectors[i][start_dim:end_dim] = v[token.strip()]
            start_dim = end_dim
        assert (start_dim == tot_dim)
```





## 总结

所有 当 `build_vocab` 之后：

* `Field` 对象中就保存了一个 `Vocab` 对象
* `Vocab` 对象保存了 当前 `corpus` 所有 `token` 的 `embedding`



## 其它

```python
# 计数 dict，顺序为 计数的大小
from collections import Counter


```



