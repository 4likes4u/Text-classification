# Text-classification

Simple text classification using [DAN](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf), LSTM and [BERT](https://arxiv.org/abs/1810.04805) for [SMP-19-ECISA](http://conference.cipsc.org.cn/smp2019/smp_ecisa_SMP.html) contest.

# Dependencies
- python 3.7
- pytorch
- transformers
- pytorch-ignite
- numpy

# Training

Already uploaded the clean version of raw text, named "train.txt", "dev.txt" and "test.txt" for DAN and LSTM, and "bert_train.txt" and "bert_dev.txt" for BERT.

To training BERT, personally, I strongly recommend you to use the gpu version code, see "bert.py" instead of "bert.ipythonotebook".

# Performance

SMP only offers label for train and dev, so I only report performance over dev set here, the metric here is accuracy.

|Model|DAN|LSTM|BERT|
| ---------- | :----------: | :-----------:  | :-----------: |
|Accuracy|  0.752   |   0.720    |   0.819 |
