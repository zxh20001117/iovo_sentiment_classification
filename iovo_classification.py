import numpy as np
from configparser import ConfigParser

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel

from generate_bert_vector import get_data

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

bert = BertModel.from_pretrained(conf.get("train", "pretrained_model"))


def get_bert_seq_vector(tokens, attention_mask):
    embeddings = bert(tokens).last_hidden_state  # shape [batch, seq_length, 768]

    # attention mask shape [batch, seq_length] ---→ mask shape [batch, seq_length, 768]
    # attention_mask :
    # tensor([
    #     [1, 1, 1, ..., 0, 0, 0],
    #     ...
    #     ...
    #     ...
    # ])
    #
    # mask:
    # tensor([[[1., 1., 1., ..., 1., 1., 1.],
    #          [1., 1., 1., ..., 1., 1., 1.],
    #          [1., 1., 1., ..., 1., 1., 1.],
    #          ...,
    #          [0., 0., 0., ..., 0., 0., 0.],
    #          [0., 0., 0., ..., 0., 0., 0.],
    #          [0., 0., 0., ..., 0., 0., 0.]],
    #         ...
    #         ...
    #         ...
    # ])
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # 每个向量表示一个单独token的掩码——现在每个token都有一个大小为768的向量，表示它的attention_mask状态。然后将两个张量相乘
    masked_embeddings = embeddings * mask

    # 沿着轴1将剩余的嵌入项求和:
    summed = torch.sum(masked_embeddings, 1)

    # 将张量的每个位置上的值相加
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)

    # 计算平均值 作为句向量
    mean_pooled = summed / summed_mask
    return mean_pooled


def get_single_ovo_score(sentence, level_one, level_two):
    return 1


def get_iovo_score(sentence):
    R = np.zeros((5, 5))
    for level_one in range(1, 6):
        for level_two in range(1, 6):
            if level_one < level_two:
                R[level_one - 1, level_two - 1] = get_single_ovo_score(sentence, level_one, level_two)
                R[level_two - 1, level_one - 1] = 1.0 - R[level_one - 1, level_two - 1]


T = np.zeros((5, conf.getint("classification", "sentence_vector_length")))

train_data = get_data(conf.get("data", "train_path"), return_attention_mask=True).drop(["sentence"], axis=1)
train_data = train_data.reset_index()

train_token = torch.tensor(np.array([i for i in train_data['token']]))
train_attention_mask = torch.tensor(np.array([i for i in train_data['attention_mask']]))
train_label = torch.tensor(np.array([i for i in train_data['sentiment']]))

train_dataset = TensorDataset(train_token, train_attention_mask, train_label)
get_vector_batch_size = conf.getint("classification", "batch_size")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=get_vector_batch_size)

train_vector = []
for token, attention_mask, label in train_loader:
    vectors = get_bert_seq_vector(token, attention_mask)
    for i in list(vectors.detach().numpy()):
        train_vector.append(i)

train_data_vector = pd.DataFrame({'vector': train_vector})
train_data = pd.concat([train_data, train_data_vector], axis=1)
train_data.to_json("data/train_sentence_vector_bert-last-hidden-state.json")
print(len(train_data), len(train_data_vector))
