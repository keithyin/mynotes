**传统 TTS 问题分为三个部分解决**

- text analysis fronted : 文本分析前端
  - 抽取音素特征，重音。等等语音特征。
- acoustic model ： 给定音素，输出声学特征（`duration prediction` , `fundamental frequency` 等等）。
- audio synthesis module ： 根据声学特征，合成声音



# Tacotron: Towards End-to-End Speech Synthesis

