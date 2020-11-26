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
        self.embed = nn.Embedding(len(field.vocab), embedding_dim=256)
        self.convs = nn.ModuleList([nn.Conv2d(1, 128, (k, 256)) for k in [3, 4, 5, 6]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4, 5)

    def forward(self, text):
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


class TextRNN(nn.Module):
    def __init__(self, field):
        super(TextRNN, self).__init__()
        self.embed = nn.Embedding(len(field.vocab), embedding_dim=256, padding_idx=len(field.vocab) - 1)
        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128 * 2, 5)

    def forward(self, text):
        x = text.transpose(0, 1)
        out = self.embed(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out


class TextRCNN(nn.Module):
    def __init__(self, field):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(len(field.vocab), embedding_dim=256, padding_idx=len(field.vocab) - 1)
        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        # 每句话处理成的长度(短填长切)
        self.maxpool = nn.MaxPool1d(268)
        # fc的输入是LSTM输出大小+embedding层的输出大小
        self.fc = nn.Linear(128 * 2 + 256, 5)

    def forward(self, text):
        x = text.transpose(0, 1)
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        return out


# class GlobalMaxPool1d(nn.Module):
#     def __init__(self):
#         super(GlobalMaxPool1d, self).__init__()
#
#     def forward(self, x):
#         return torch.max_pool1d(x, kernel_size=x.shape[-1])
#
#
# class TextRCNN(nn.Module):
#     def __init__(self, field):
#         super(TextRCNN, self).__init__()
#         self.embed = nn.Embedding(len(field.vocab), 256)
#         self.lstm = nn.LSTM(input_size=256, hidden_size=128,
#                             batch_first=True, bidirectional=True)
#         self.globalmaxpool = GlobalMaxPool1d()
#         self.dropout = nn.Dropout(.5)
#         self.linear1 = nn.Linear(256 + 2 * 128, 256)
#         self.linear2 = nn.Linear(256, 5)
#
#     def forward(self, x):  # x: [batch,L]
#         x_embed = self.embed(x)  # x_embed: [batch,L,embedding_size]
#         last_hidden_state, (c, h) = self.lstm(x_embed)  # last_hidden_state: [batch,L,hidden_size * num_bidirectional]
#         out = torch.cat((x_embed, last_hidden_state),
#                         2)  # out: [batch,L,embedding_size + hidden_size * num_bidirectional]
#         # print(out.shape)
#         out = F.relu(self.linear1(out))
#         out = out.permute(dims=[0, 2, 1])  # out: [batch,embedding_size + hidden_size * num_bidirectional,L]
#         out = self.globalmaxpool(out).squeeze(-1)  # out: [batch,embedding_size + hidden_size * num_bidirectional]
#         # print(out.shape)
#         out = self.dropout(out)  # out: [batch,embedding_size + hidden_size * num_bidirectional]
#         out = self.linear2(out)  # out: [batch,num_labels]
#         return out


class TextRNN_Atten(nn.Module):
    def __init__(self, field):
        super(TextRNN_Atten, self).__init__()
        self.embedding = nn.Embedding(len(field.vocab), embedding_dim=256, padding_idx=len(field.vocab) - 1)
        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(128 * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc = nn.Linear(64, 5)

    def forward(self, text):
        x = text.transpose(0, 1)
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out