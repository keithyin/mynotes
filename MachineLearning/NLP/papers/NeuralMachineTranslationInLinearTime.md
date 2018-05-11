# NeuralMachineTranslationInLinearTime

> it runs in time that is linear in the length of the sequences and it sidesteps the need for excessive memorization



> We find that the latent alignment structure contained in the representations reflects the expected alignment between the tokens.



> The larger the distance, the harder it is to learn the dependencies between the tokens



**提出的网络具有两种性质：**

* > it runs in time that is linear in the length of the sequences

* > it sidesteps the need for excessive memorization

  * 意思就是**不将**整个 source-sentence 编码成一个固定长度的向量。因为这样的话，固定向量的**压力就会非常大**, 它需要记住非常多的东西。
  * ​