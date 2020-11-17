import numpy as np
import random
import torch
import os
import copy
import time
import datetime
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.':
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return sentences, target


SEED = 123
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "./dataset/"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 读取数据，是由tuple组成的列表形式
sentences, target = load_data(path)
target = torch.tensor(target)

# BertTokenizer进行编码，将每一句转成数字
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./results")
print(sentences[2])
print(tokenizer.tokenize(sentences[2]))
print(tokenizer.encode(sentences[2]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sentences[2])))


# 将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=200):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]

input_tokens = torch.tensor(input_ids)
print("token shape:", input_tokens.shape)  # torch.Size([10000, 128])


# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)


train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, target, random_state=666, test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.2)
print(train_inputs.shape, test_inputs.shape)      #torch.Size([8000, 128]) torch.Size([2000, 128])
print(train_masks.shape)                          #torch.Size([8000, 128])和train_inputs形状一样

print(train_inputs[0])
print(train_masks[0])

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# 创建模型、loss和优化器
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels = 5)     # num_labels表示5个分类
model.to(device)

# 通用的写法：bias和LayNorm.weight没有用权重衰减
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = LEARNING_RATE, eps = EPSILON)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))   #返回 hh:mm:ss 形式的时间


def cal_acc(preds, labels):      # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()      #eq里面的两个参数的shape=torch.Size([16])
    acc = correct.sum().item() / len(correct)
    return acc


num_epoches = 50
best_acc = 0
best_model_wts = copy.deepcopy(model.state_dict())
model.train()
avg_train_loss, avg_train_acc = [], []
avg_val_loss, avg_val_acc = [], []

# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_data) * num_epoches
# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

print('Start training...')
start = time.time()
for epoch in range(num_epoches):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)

        output = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]
        train_loss += loss.item() * len(batch)
        acc = cal_acc(logits, b_labels)
        train_acc += acc * len(batch)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)      #大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()              #更新模型参数
        scheduler.step()              #更新learning rate

    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)

            output = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask, labels=b_labels)
            loss, logits = output[0], output[1]
            val_loss += loss.item() * len(batch)
            acc = cal_acc(logits, b_labels)
            val_acc += acc * len(batch)

    train_acc /= len(train_data)
    train_loss /= len(train_data)
    val_acc /= len(test_data)
    val_loss /= len(test_data)
    avg_train_loss.append(train_loss)
    avg_train_acc.append(train_acc)
    avg_val_loss.append(val_loss)
    avg_val_acc.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print('Epoch: {} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch + 1, train_loss, train_acc))
    print('\t\t Val Loss: {:.4f} Val Acc: {:.4f}'.format(val_loss, val_acc))
    torch.cuda.empty_cache()

end = time.time()
print('Training cost %.2f mins.' % ((end - start) / 60.))
print('Best val acc: %s' % best_acc)

model.load_state_dict(best_model_wts)
timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))
model_name = 'bert_v2_%s' % (timeStr)
torch.save(model, './results/%s.pth' % model_name)

# 绘制曲线
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(avg_train_acc)
plt.plot(avg_val_acc)
plt.legend(('Train acc', 'Val acc'))
plt.title("Acc")
plt.subplot(122)
plt.plot(avg_train_loss)
plt.plot(avg_val_loss)
plt.title("Loss")
plt.legend(('Train loss', 'Val loss'))
plt.savefig('./results/%s.jpg' % model_name)