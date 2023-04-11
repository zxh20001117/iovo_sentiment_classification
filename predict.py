import pickle
import threading
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
from iovo_classification import cal_center_distance, cal_k_nearest_avg_distance, cal_relative_competence_weight

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')
USE_CUDA = torch.cuda.is_available()

d2v_model = None

# 需要加载doc2vecmodel 则改为True
if True:
    doc2vec_model = Doc2Vec.load(conf.get("doc2vec", "modelPath")+f"/{conf.get('iovo', 'doc2vec_model')}")

def get_sentences_list(df):
    sentences = []
    with open("data/value_attributes.pickle", "rb") as f:
        attribute_group = pickle.load(f)
    for values in attribute_group.keys():
        for line in df[f"{values}_sentences"]:
            for s in line:
                sentences.append(s)
    return sentences


def predict_ovo(data_loader, level_one, level_two):
    model = bert_lstm().cuda()
    model.load_state_dict(torch.load(f"{conf.get('train', 'modelPath')}/{level_one} vs {level_two} bert-lstm.pth")['model'])
    model.eval()

    predict = []
    for inputs in data_loader:
        h = model.init_hidden(inputs.size(0))
        if USE_CUDA:
            inputs = inputs.cuda()
        h = tuple([each.data for each in h])
        output = model(inputs, h)
        probs = torch.softmax(output, dim=1)
        true_probs = probs[:, 1].tolist()
        predict = predict + true_probs
    return predict


def predict_all_ovo(sentences):
    tokenizer = BertTokenizer.from_pretrained(conf.get("train", "pretrained_model"))
    token_data = tokenizer.batch_encode_plus(
        sentences.sentence.values,
        add_special_tokens=conf.getboolean("train", "add_special_tokens"),
        return_attention_mask=conf.getboolean("train", "return_attention_mask"),
        padding='max_length',
        max_length=conf.getint("train", "seq_length"),
        return_tensors=conf.get("train", "return_tensors"),
        truncation=True
    )
    temp_sentences = sentences
    # tensorData = TensorDataset()
    batch_size = conf.getint("predict", "batch_size")
    data_loader = DataLoader(token_data['input_ids'], shuffle=False, batch_size=batch_size, drop_last=False)
    for level_one in range(1, 6):
        for level_two in range(level_one + 1, 6):
            predict = predict_ovo(data_loader, level_one, level_two)
            temp_sentences[f"{level_one}-{level_two}"] = predict


def get_iovo_class(sentences):
    predicts = []
    count = 0
    for i in range(len(sentences)):
        R = np.zeros((5, 5))
        for level_one in range(1, 6):
            for level_two in range(1, 6):
                if level_one < level_two:
                    R[level_one - 1, level_two - 1] = sentences.iloc[i][f'{level_one}-{level_two}']
                    R[level_two - 1, level_one - 1] = 1.0 - R[level_one - 1, level_two - 1]
        d2v_vector = doc2vec_model.infer_vector(word_tokenize(sentences.iloc[i]['sentence']))
        dC = cal_center_distance(d2v_vector)
        dK = cal_k_nearest_avg_distance(d2v_vector)
        weighted_marix = cal_relative_competence_weight(dC, dK)

        weighted_R = R * weighted_marix
        row_sums = weighted_R.sum(axis=1)
        most_likely_class = row_sums.argmax() + 1
        predicts.append(most_likely_class)
        count +=1
        if count %10 == 0: print(count)
    sentences['IOVO predict'] = predicts


class IOVOThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        print(f'当前子进程：{threading.currentThread().name}')
        self.func(self.args[0])
        print(f'子进程：{threading.currentThread().name} 执行完成')


if __name__ == "__main__":
    level = "middle"

    # 生成每一句话的 十个 ovo 分数 保存为 level hotel scores.json 文件

    # data = pd.read_json(f"data/{level} level hotel data.json")
    # sentences = pd.DataFrame(get_sentences_list(data), columns=['sentence'])
    # predict_all_ovo(sentences)
    # sentences.to_json(f"result/{level} hotel ovo scores.json")
    # print(sentences.columns)


    # 计算 IOVO值 保存为 level hotel iovo scores.json
    # sentences = pd.read_json(f"result/{level} hotel ovo scores.json")
    # nGroup = 6
    # sentences_groups = []
    # length = len(sentences)//nGroup
    # for i in range(nGroup - 1):
    #     sentences_groups.append(sentences.iloc[i*length:(i+1)*length].copy())
    # sentences_groups.append(sentences.iloc[(nGroup-1)*length:].copy())
    # del sentences
    # threadList = []
    # for index, t in enumerate(sentences_groups):
    #     iT = IOVOThread(get_iovo_class, (sentences_groups[index], ))
    #     threadList.append(iT)
    # for m in threadList:
    #     m.start()
    # for m in threadList:
    #     m.join()
    #
    # input("请等待所有子进程结束，之后按任意键继续：")
    #
    # result = pd.concat(sentences_groups)
    # result.to_json(f"result/{level} hotel iovo scores.json")
    # # get_iovo_class(sentences)






