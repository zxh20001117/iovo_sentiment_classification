import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from configparser import ConfigParser
import torch.nn.functional as F

from generate_bert_vector import data_split, get_data, get_iovo_data

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')
USE_CUDA = torch.cuda.is_available()


class bert_lstm(nn.Module):
    def __init__(self,
                 hidden_dim=conf.getint("train", "hidden_dim"),
                 output_size=conf.getint("train", "output_size"),
                 n_layers=conf.getint("train", "n_layers"),
                 bidirectional=conf.getboolean("train", "bidirectional"),
                 drop_prob=conf.getfloat("train", "drop_prob")):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert = BertModel.from_pretrained(conf.get("train", "pretrained_model"))
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim,
                            conf.getint("train", "n_layers"),
                            batch_first=True,
                            bidirectional=conf.getboolean("train", "bidirectional")
                            )

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)


        # linear and sigmoid layers
        if bidirectional:
            self.fc1 = nn.Linear(hidden_dim * 2, conf.getint("train", "full_con1_out"))
        else:
            self.fc1 = nn.Linear(hidden_dim, conf.getint("train", "full_con1_out"))

        self.act = nn.ReLU()

        self.fc2 = nn.Linear(conf.getint("train", "full_con1_out"), 2)

        # self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
    def forward(self, x, hidden):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert 字向量

        # lstm_out
        # x = x.float()
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)
        # print(lstm_out.shape)   #[32,100,768]
        # print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc1(out)
        out = self.act(out)

        out = self.fc2(out)
        return out

        # out = F.softmax(out, dim=-1)
        # return out


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )

        return hidden


if __name__ == "__main__":
    model = bert_lstm()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.getfloat("train", "lr"))

    if USE_CUDA: model.cuda()

    model.train()

    # train for some number of epochs
    epochs = conf.getint("train", "epochs")
    data = get_data("sentimentData/marked sentences sentiment 2.xlsx")
    data = get_iovo_data(data, 1, 2)
    train_loader, valid_loader, test_loader = data_split(data)

    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(conf.getint("train", "batch_size"))
        counter = 0
        print_every = conf.getint("train", "print_every")
        batch_size = conf.getint("train", "batch_size")

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if USE_CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])
            model.zero_grad()
            output = model(inputs, h)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_h = model.init_hidden(batch_size)
                    val_losses = []
                    for inputs, labels in valid_loader:
                        val_h = tuple([each.data for each in val_h])

                        if (USE_CUDA):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = model(inputs, val_h)
                        val_loss = criterion(output, labels)

                        val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
