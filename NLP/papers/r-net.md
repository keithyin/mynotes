# R-Net

> * We first match the question and passage with gated attention-based recurrent networks to obtain the **question-aware passage representation**. 
> * Then we propose a self-matching attention mechanism to refine the representation by matching the passage against itself, which effectively encodes information from the whole passage. 
> * We finally employ the **pointer networks** to locate the positions of answers from the passages.



>  Our model consists of four parts: 
>
> * the recurrent network encoder to build representation for questions and passages **separately**,
> * the **gated matching layer** to match the question and passage
> * the **self-matching layer** to aggregate information from the whole passage
> * the **pointer-network** based answer boundary prediction layer. 



**论文中符号**

$Q = \{w_t^Q \}^m_{t=1}$ : $w$ 代表 $word$ 意思, $m$ 说明这个 Question 有 m 个单词.

$P = \{w_t^P \}^n_{t=1}$ : $n$ 表示 Passage 有 $n$ 个单词

$\{e_t^Q\}_{t=1}^m$ :  Question 的 word-embedding 表示

$\{e_t^P\}_{t=1}^n$ :  Passage 的 word-embedding 表示

$\{c_t^Q\}_{t=1}^m$ :  Question 的 characters-embedding 表示

$\{c_t^P\}_{t=1}^n$ :  Passage 的 characters-embedding 表示

