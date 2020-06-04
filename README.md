# Text-classification

Simple text classification using DAN, LSTM and BERT for SMP-19-ECISA contest.

# Dependency
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

|DAN|LSTM|BERT|
| ---------- | :-----------:  | :-----------: |
|  0.752   |   0.720    |   0.819 |
