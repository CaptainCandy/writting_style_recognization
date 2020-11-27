###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
# 导入相关包
import copy
import os
import time
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

from mymodel import MyNet, TextCNN, TextRNN, TextRCNN, TextRNN_Atten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "./dataset/"

# 精确模式分词
# titles = [".".join(jb.cut(t, cut_all=False)) for t,_ in dataset.items()]
# print("精确模式分词结果:\n",titles[0])
# # 全模式分词
# titles = [".".join(jb.cut(t, cut_all=True)) for t,_ in dataset.items()]
# print("全模式分词结果:\n",titles[0])
# # 搜索引擎模式分词
# titles = [".".join(jb.cut_for_search(t)) for t,_ in dataset.items()]
# print("搜索引擎模式分词结果:\n",titles[0])

# 将片段进行词频统计
def tfidf(path):
    dataset = {}
    files= os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.': # 跳过隐藏文件和文件夹
            f = open(path+"/"+file, 'r', encoding='UTF-8'); # 打开文件
            for line in f.readlines():
                dataset[line] = file[:-4]
    name_zh = {'LX': '鲁迅', 'MY':'莫言' , 'QZS':'钱钟书' ,'WXB':'王小波' ,'ZAL':'张爱玲'}
    for (k,v) in  list(dataset.items())[:6]:
        print(k,'---',name_zh[v])
    str_full = {}
    str_full['LX'] = ""
    str_full['MY'] = ""
    str_full['QZS'] = ""
    str_full['WXB'] = ""
    str_full['ZAL'] = ""

    for (k,v) in dataset.items():
        str_full[v] += k

    for (k,v) in str_full.items():
        print(k,":")
        for x, w in jb.analyse.extract_tags(v, topK=5, withWeight=True):
            print('%s %s' % (x, w))

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者
    
    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    # max_len = 0
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.':
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])
                # if len(line) > max_len:
                #     max_len = len(line)

    return list(zip(sentences, target))

# 定义Field
# Field可以理解为特定的文本数据类型
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
              lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path)

# 使用Example把定义好的field应用到每一条记录中去，构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)
# tfidf(path)

max_len = 0
for t in dataset:
    if len(t.text) > max_len:
        max_len = len(t.text)

# 切分数据集
train, val = dataset.split(split_ratio=0.7)

# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (train, val), # 数据集
    batch_sizes=(8, 8),
    device=device, # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    repeat=False
)

print("Max length: %s" % max_len)
print(train.examples[0].text[0], train.examples[0].category)
print('Finish loading data on %s.' % device)

#%%
# 创建模型实例
# model = TextCNN(TEXT)
# model = TextRNN(TEXT)
model = TextRCNN(TEXT)
# model = TextRNN_Atten(TEXT)
model = model.to(device)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []

# train
num_epoches = 50
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

print('Start training...')
start = time.time()
for epoch in range(num_epoches):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0

    for idx, batch in enumerate(train_iter):
        text, label = batch.text, batch.category
        optimizer.zero_grad()
        out = model(text)
        loss = loss_fn(out,label.long())
        loss.backward(retain_graph=True)
        optimizer.step()
        accuracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
        # 计算每个样本的acc和loss之和
        train_acc += accuracy*len(batch)
        train_loss += loss.item()*len(batch)
        
        # print("\r epoch:{} loss:{}, train_acc:{}".format(epoch+1, loss.item(), accracy),end=" ")
        
    
    # 在验证集上预测
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, label = batch.text, batch.category
            out = model(text)
            loss = loss_fn(out,label.long())
            accuracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
            # 计算一个batch内每个样本的acc和loss之和
            val_acc += accuracy*len(batch)
            val_loss += loss.item()*len(batch)
    
    train_acc /= len(train_iter.dataset)
    train_loss /= len(train_iter.dataset)
    val_acc /= len(val_iter.dataset)
    val_loss /= len(val_iter.dataset)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print('Epoch: {} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch + 1, train_loss, train_acc))
    print('\t\t Val Loss: {:.4f} Val Acc: {:.4f}'.format(val_loss, val_acc))

end = time.time()
print('Training cost %.2f mins.' % ((end - start) / 60.))
print('Best val acc: %s' % best_acc)

# 保存模型
model.load_state_dict(best_model_wts)
timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))
model_name = 'TextRCNN_v1_%s' % (timeStr)
torch.save(model, './results/%s.pth' % model_name, _use_new_zipfile_serialization=False)

# 绘制曲线
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title("Acc")
plt.legend(('Train acc', 'Val acc'))
plt.subplot(122)
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title("Loss")
plt.legend(('Train loss', 'Val loss'))
plt.savefig('./results/%s.jpg' % model_name)