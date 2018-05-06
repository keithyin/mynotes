# pytorch: pad pack sequence

在使用 pytorch 的 RNN 模块的时候, 有时会不可避免的使用到 `pack_padded_sequence` 和 `pad_packed_sequence`,  **当使用双向RNN的时候, 必须要要使用 pack_padded_sequence !!** .否则的话, pytorch 是无法获得 序列的长度, 这样也无法正确的计算双向 `RNN/GRU/LSTM` 的结果.

但是在使用 `pack_padded_sequence` 时有个问题, 即输入 mini-batch 序列的长度必须是从长到短排序好的, 当mini-batch 中的样本的顺序非常的重要的话, 这就有点棘手了. 比如说, 每个 sample 是个 单词的 字母级表示, 一个 mini-batch 保存了一句话的 words. 

在这种情况下, 我们依然要使用 `pack_padded_sequence`, 所以需要先将 mini-batch 中样本排序, 然后 `RNN/LSTM/GRU` 计算完之后再恢复成以前的顺序.  

下面的代码将用来实现这种方法:

```python
import torch
from torch import nn
from torch.autograd import Variable

def func():
    bs = 3
    max_time_step = 5
    feat_size = 15
    hidden_size = 7
    seq_lengths = [3, 5, 2]

    rnn = nn.GRU(input_size=feat_size, 
                 hidden_size=hidden_size, batch_first=True, bidirectional=True)
    x = Variable(torch.FloatTensor(bs, max_time_step, feat_size).normal_())
	
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    sorted_seq_lengths, indices = torch.sort(
        torch.LongTensor(seq_lengths), descending=True)
    
    # 如果我们想要将计算的结果恢复排序前的顺序的话, 
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],  
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)
    
    # 对原始序列进行排序
    sorted_x = x[indices]
    packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, 
                          sorted_seq_lengths.numpy(), batch_first=True)

    res, state = rnn(packed_x)

    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
	
    # 恢复排序前的样本顺序
    desorted_res = padded_res[desorted_indices]
    print(desorted_res)
	
    # 不使用 pack_paded, 用来和上面的结果对比一下.
    not_packed_res, _ = rnn(x)
    print(not_packed_res)
```

