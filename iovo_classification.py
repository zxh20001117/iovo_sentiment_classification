import time

import numpy as np
from configparser import ConfigParser

import pandas as pd
import torch
from gensim.models import Doc2Vec
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer
from nltk.tokenize import word_tokenize

from classification_train import bert_lstm
from generate_bert_vector import get_data

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

bert = BertModel.from_pretrained(conf.get("train", "pretrained_model"))
tokenizer = BertTokenizer.from_pretrained(conf.get("train", "pretrained_model"))
modelDict = {}
d2v_model = None

# 需要加载doc2vecmodel 则改为True
if True:
    doc2vec_model = Doc2Vec.load(conf.get("doc2vec", "modelPath")+"/d2v.model")



try:
    train_data = pd.read_json("data/train_sentence_vector_doc2vec.json").drop(
        ['index', 'Unnamed: 0'],
        axis=1)
    group_data = train_data.groupby('sentiment')
    avg_vectors = {"label": [], "avg_vector": []}
    for i in group_data.groups.keys():
        temp_data = group_data.get_group(i)
        group_vectors = np.array([j for j in temp_data['vector']])
        avg_vector = np.mean(group_vectors, axis=0)
        avg_vectors['label'].append(i)
        avg_vectors['avg_vector'].append(avg_vector)
    avg_vectors = pd.DataFrame(avg_vectors)

except FileNotFoundError:
    print("no train vectors, please use function generate_train_vector first")
    exit(0)


def load_basic_models(modelDict):
    for i in range(1, 6):
        for j in range(i+1, 6):
            model = bert_lstm()
            model.load_state_dict(torch.load(f"{conf.get('train', 'modelPath')}/{i} vs {j} bert-lstm.pth"))
            model.eval()
            modelDict[f"{i} vs {j}"] = model


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def cal_center_distance(vector_in):
    distance = get_cos_similar_multi(vector_in, list(avg_vectors['avg_vector']))[0]
    return distance


def cal_k_nearest_avg_distance(vector_in, K = 5):
    distance = np.zeros(5)
    for i in group_data.groups.keys():
        temp_data = group_data.get_group(i)
        arr = get_cos_similar_multi(vector_in, list(temp_data['vector']))[0]
        max_indexes = np.argsort(arr)[-K:]  # 获取最大的K个元素的索引
        distance[i - 1] = arr[max_indexes].mean()  # 通过索引获取最大的K个元素
    return distance


def cal_relative_competence_weight(dC, dK):
    weight_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            weight_matrix[i][j] = (dC[i] * dC[i] / (dC[i] * dC[i] + dC[j] * dC[j])) * \
                                  (dK[i] * dK[i] / (dK[i] * dK[i] + dK[j] * dK[j]))
    for i in range(5):
        weight_matrix[i][i] = 0
    return weight_matrix


def get_bert_seq_vector(tokens, attention_mask, vector_model= 'mean_pooled'):
    if vector_model == 'mean_pooled':
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

    elif vector_model == 'pooler_output':
        pooler_output = bert(tokens)[-1]
        return pooler_output


def get_single_ovo_score(token, level_one, level_two):
    h = modelDict[f"{level_one} vs {level_two}"].init_hidden(conf.getint("train", "batch_size"))
    output = modelDict[f"{level_one} vs {level_two}"](token, h)
    probs = torch.softmax(output, dim=1)
    true_probs = probs[:, 1].tolist()
    return true_probs[0]



def get_iovo_R_matrix(sentence):
    R = np.zeros((5, 5))
    token = tokenizer.batch_encode_plus(
        sentence,
        add_special_tokens=conf.getboolean("train", "add_special_tokens"),
        return_attention_mask=conf.getboolean("train", "return_attention_mask"),
        padding='max_length',
        max_length=conf.getint("train", "seq_length"),
        return_tensors=conf.get("train", "return_tensors")
    )

    for level_one in range(1, 6):
        for level_two in range(1, 6):
            if level_one < level_two:
                R[level_one - 1, level_two - 1] = get_single_ovo_score(token, level_one, level_two)
                R[level_two - 1, level_one - 1] = 1.0 - R[level_one - 1, level_two - 1]
    return R

def get_iovo_class_result(sentence):
    R = get_iovo_R_matrix(sentence)
    d2v_vector = doc2vec_model.infer_vector(word_tokenize(sentence))
    dC = cal_center_distance(d2v_vector)
    dK = cal_k_nearest_avg_distance(d2v_vector)
    weighted_marix = cal_relative_competence_weight(dC, dK)
    weighted_R = R * weighted_marix
    row_sums = weighted_R.sum(axis=1)
    most_likely_class = row_sums.argmax() + 1
    return most_likely_class


T = np.zeros((5, conf.getint("classification", "sentence_vector_length")))


def generate_train_vector(vector_model = 'mean_pooled'):
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
        vectors = get_bert_seq_vector(token, attention_mask, vector_model)
        for i in list(vectors.detach().numpy()):
            train_vector.append(i)

    train_data_vector = pd.DataFrame({'vector': train_vector})
    train_data = pd.concat([train_data, train_data_vector], axis=1)
    if vector_model == 'mean_pooled':
        train_data.to_json("data/train_sentence_vector_bert-last-hidden-state.json")
    elif vector_model == 'pooler_output':
        train_data.to_json("data/train_sentence_vector_bert-pooler-output.json")


def generate_train_doc2vec_vector():
    startTime = time.time()
    df = pd.read_excel(conf.get("data", "train_path")).reset_index()
    loadTime = time.time()
    print(f'it takes {loadTime - startTime:.1f} s for loading data\n')
    doc2vec_vectors = [doc2vec_model.infer_vector(doc.split()) for doc in df['sentence']]
    df['vector'] = doc2vec_vectors
    generateTime = time.time()
    print(f'it takes {generateTime - loadTime:.1f} s for generating vectors\n')
    print(df['vector'].head())
    df.to_json('data/train_sentence_vector_doc2vec.json')




if __name__ == '__main__':
    test_sentence = "i had a very uncomfortable night in that room and therefore i went to the front desk the next day to request a serious change of room"
    test_vector = doc2vec_model.infer_vector(test_sentence.split())
    print(test_sentence)
    dC = cal_center_distance(test_vector)
    print(dC)
    dK = cal_k_nearest_avg_distance(test_vector, K=5)
    print(dK)
    w = cal_relative_competence_weight(dC, dK)
    print(w)
    row_sums = w.sum(axis=1)
    most_likely_class = row_sums.argmax() + 1
    print("most likely class: ", most_likely_class)

