#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import numpy as np
from torch import nn
import torch.utils.data as Data
from transformers import BertModel


# In[6]:


# 读取数据
train = np.loadtxt("bert_train.txt")
dev = np.loadtxt("bert_dev.txt")
#test = np.loadtxt("test.txt")


# In[7]:


print(train.shape) #14788, 101
print(dev.shape)# 5145, 101


# In[8]:


from ignite.metrics import Accuracy
def eval_at_dev(model, dev_iter):
    acc = Accuracy()
    for sent, gd_label in dev_iter:
        pred = model(sent.cuda(9)) #get model predict
        acc.update((pred, gd_label.cuda(9)))
    acc_rate = acc.compute()
    print("current model over dev set accuracy rate: " + str(acc_rate))
    return acc_rate


# In[9]:


#定义数据迭代器
batch_size = 100
train_set = Data.TensorDataset(torch.LongTensor(train[:, 1:103]), torch.LongTensor(train[:, 0]))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)

dev_batch_size = 49
dev_set = Data.TensorDataset(torch.LongTensor(dev[:, 1:103]), torch.LongTensor(dev[:, 0]))
dev_iter = Data.DataLoader(dev_set, dev_batch_size, shuffle=False)


# In[13]:


#定义模型


class BertClassifier(nn.Module):

    def __init__(self):
        super(BertClassifier, self).__init__()
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # classifier by a single fc layer
        self.fc = nn.Linear(768, 3)
        # Weight initialization especially for fc layer, bert shouldn't be initialized twice
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # compute by pre-trained BERT
        outputs = self.bert(x)
        # only need last layer output (Total 12 layers)
        output = outputs[-1]
        res = self.fc(output)
        return res

class LSTM(nn.Module): 
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(5895, 100) #5895 个词，每个词的嵌入是100维
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)#定义LSTM，LSTM的输入是(batch_size, seq_len, embedding_size)
        #注意LSTM的输出有两个，我们只要第一个
#         self.dropout = nn.Dropout(0.2) # 后期我们会详述嘛是dropout
        self.fc1 = nn.Linear(100, 300)
        self.fc2 = nn.Linear(300, 3)

    def forward(self, x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        x = x.mean(dim=1) #为了节省空间，依旧采用比较简单的mean 操作
#         x = self.dropout(x)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x

class DAN(nn.Module): 
    def __init__(self):
        super(DAN, self).__init__()
        self.embedding = nn.Embedding(5895, 100) #5895 个词，每个词的嵌入是100维
        self.fc1 = nn.Linear(100, 300)
        self.fc2 = nn.Linear(300, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
#         x = self.dropout(x)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x
#实例化
model = BertClassifier().cuda(9)


# In[14]:


#优化步骤

#定义Loss, 我们使用softmax-cross-entropy-loss
loss = nn.CrossEntropyLoss()

#定义优化算法
import torch.optim as optim #
optimizer = optim.Adam(model.parameters(), lr=5e-5) #使用Adam 优化器


# In[15]:


#进行训练
max_epoch = 10
print("before training dev accuracy:")
eval_at_dev(model, dev_iter)
print("let's start training")
for epoch in range(max_epoch):
    for sent, label in train_iter:
        output = model(sent.long().cuda(9))
        L = loss(output, label.long().cuda(9))
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
    eval_at_dev(model, dev_iter)
    print('epoch: %d, loss: %f' % (epoch, L.item()))


# In[ ]:



