import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()                
        self.lstm = torch.nn.LSTM(1,64)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,5)

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output, hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1],-1)))
        return out


class TextCNN(nn.Module):
    def __init__(self, field):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(len(field.vocab), 256)
        self.convs = nn.ModuleList([nn.Conv2d(1, 128, (k, 256)) for k in [3, 4, 5, 6]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4, 5)

    def forward(self, text):
        '''

        :param text: 输入
        :return: 输出
        '''

        x = text.transpose(0, 1)
        x = self.embed(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(l, l.size(2)).squeeze(2) for l in x]

        x = torch.cat(x, 1)
        # print("x shape: ", x.shape)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits
