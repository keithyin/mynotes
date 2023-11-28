

## fastq, fastqc

fastq: 每个序列有四行数据  (reads + quality = fast q file)
1. Sequence ID
2. Sequence
3. Quality ID
4. Quality score. Phred quality. Ascii of: sequence quality + 33, -10 log10 pr (sequnece quality)
    1. 如果知道了质量不好怎么办？重新测一遍？质量不好的地方进行截断？ 或者直接丢弃？
    2. 

```
@HWI_EAS305:1:1:1:101#0/1
ATCGGTCA
+HWI_EAS305:1:1:1:101#0/1
WVXUWVRK
@HWI_EAS305:1:1:1:102#0/1
ATCGGTCA
+HWI_EAS305:1:1:1:102#0/1
WVXUWVRK
```

fastqc: per base sequence quality (如果有一堆序列，看每个位置的 quality 值分布。主要是关注位点的 质量)

## sequence mapping

1. check the sequence quality is ok
2. map the sequence to the reference genome
