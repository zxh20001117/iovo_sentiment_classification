import random
from configparser import ConfigParser
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from classification_train import bert_lstm
from generate_bert_vector import get_data, get_iovo_data, data_split
from iovo_classification import cal_relative_competence_weight
from predict import predict_all_ovo, get_iovo_class, IOVOThread, get_dK, get_dC

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')
USE_CUDA = torch.cuda.is_available()

random.seed(conf.getint("train", "seed_val"))
np.random.seed(conf.getint("train", "seed_val"))
torch.manual_seed(conf.getint("train", "seed_val"))
torch.cuda.manual_seed_all(conf.getint("train", "seed_val"))


def cal_TP_TN_FP_FN(preds, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(preds)):
        if preds[i] == 1 and labels[i] == 1:
            TP += 1
        if preds[i] == 0 and labels[i] == 0:
            TN += 1
        if preds[i] == 1 and labels[i] == 0:
            FP += 1
        if preds[i] == 0 and labels[i] == 1:
            FN += 1
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return TP, TN, FP, FN, Recall, Precision, F1



def split_valid_data(path):
    data = get_data(conf.get("data", "train_path"))
    data_list = []
    for i in range(1, 6):
        for j in range(1, i):
            ovo_data = get_iovo_data(data, j, i)
            ovo_data = ovo_data.copy().reset_index()
            data_train, data_test = train_test_split(ovo_data,
                                                     test_size=conf.getfloat("data_split", "test_size"),
                                                     shuffle=True,
                                                     stratify=ovo_data['label'].values,
                                                     random_state=conf.getint("data_split", "random_state"))

            data_valid, data_test = train_test_split(data_test,
                                                     test_size=conf.getfloat("data_split", "valid_size"),
                                                     shuffle=True,
                                                     stratify=data_test['label'].values,
                                                     random_state=conf.getint("data_split", "random_state"))
            data_list.append(data_valid)
            print(f"size of valid data for {j} vs {i} is {data_valid.shape[0]}")
    valid_data = pd.concat(data_list)
    valid_data.reset_index(inplace=True, drop=True)
    valid_data.to_json(path)


def cal_iovo(sentences, K=5):
    nGroup = 6
    sentences_groups = []
    length = len(sentences)//nGroup
    for i in range(nGroup - 1):
        sentences_groups.append(sentences.iloc[i*length:(i+1)*length].copy())
    sentences_groups.append(sentences.iloc[(nGroup-1)*length:].copy())
    del sentences
    threadList = []
    for index, t in enumerate(sentences_groups):
        iT = IOVOThread(get_iovo_class, (sentences_groups[index], K))
        threadList.append(iT)
    for m in threadList:
        m.start()
    for m in threadList:
        m.join()

    input("请等待所有子进程结束，之后按任意键继续：")

    result = pd.concat(sentences_groups)
    return result


def get_analysis_data(K=5):
    if not os.path.exists(conf.get("data", "analysis_path")):
        split_valid_data(conf.get("data", "analysis_path"))
    else:
        return

    data = pd.read_json(conf.get("data", "analysis_path"))

    predict_all_ovo(data)
    data = cal_iovo(data, K)
    data.to_json(conf.get("data", "analysis_path"))


def get_train_valid_result():
    if not os.path.exists(conf.get("data", "BERT_LSTM_classification_valid_analysis")):
        model_result_list = []
        for level_two in range(1, 6):
            for level_one in range(1, level_two):
                preds = []
                true_label = []
                data = get_data(conf.get("data", "train_path"))
                data = get_iovo_data(data, level_one, level_two)
                train_loader, valid_loader, test_loader = data_split(data)
                model = bert_lstm().cuda()
                model.load_state_dict(
                    torch.load(f"{conf.get('train', 'modelPath')}/{level_one} vs {level_two} bert-lstm.pth")['model'])
                model.eval()
                for inputs, labels in test_loader:
                    h = model.init_hidden(inputs.size(0))
                    h = tuple([each.data for each in h])
                    if (USE_CUDA):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model(inputs, h)
                    output = torch.nn.Softmax(dim=1)(output)
                    pred = torch.max(output, 1)[1]  # 找到概率最大的下标
                    preds.extend(pred.cpu().numpy())
                    true_label.extend(labels.cpu().numpy())
                TP, TN, FP, FN, Recall, Precision, F1 = cal_TP_TN_FP_FN(preds, true_label)
                model_result_list.append({
                    "ovo class": f"{level_one} vs {level_two}",
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "Recall": Recall,
                    "Precision": Precision,
                    "F1": F1,
                })

        model_result = pd.DataFrame(model_result_list)
        model_result.to_csv("result/BERT LSTM valid data predict analysis.csv")


def cal_dC_dK(K=5):
    data = pd.read_json(conf.get("data", "analysis_path"))
    flag = False
    if "dK" not in data.columns:
    # if True:
        print("cal dK dC")
        data["dK"] = get_dK(data, K)
        flag = True
    if "dC" not in data.columns:
        data["dC"] = get_dC(data)
        flag = True
    if flag:
        data.to_json(conf.get("data", "analysis_path"))
        print("cal dK dC done")


def cal_most_likely_class_by_ovo():
    data = pd.read_json(conf.get("data", "analysis_path"))
    flag = False
    if "ovo_class" not in data.columns:
    # if True:
        print("cal most likely class by ovo")
        flag = True
        ovo_list = []
        for i in range(len(data)):
            R = np.zeros((5, 5))
            for level_one in range(1, 6):
                for level_two in range(1, 6):
                    if level_one < level_two:
                        R[level_one-1, level_two-1] = data.loc[i, f"{level_one}-{level_two}"]
                        R[level_two - 1, level_one - 1] = 1.0 - R[level_one - 1, level_two - 1]
            ovo_list.append(np.argmax(np.sum(R, axis=1))+1)
        data["ovo_class"] = ovo_list
    if flag:
        data.to_json(conf.get("data", "analysis_path"))
        print("cal most likely class by ovo finished")


def cal_most_likely_class_by_dcdk():
    data = pd.read_json(conf.get("data", "analysis_path"))
    flag = False
    if "dC_dK_class" not in data.columns:
    # if True:
        print("cal most likely class by dC_dK")
        flag = True
        dC_dK_list = []
        for i in range(len(data)):
            weighted_marix = cal_relative_competence_weight(data.loc[i, "dC"], data.loc[i, "dK"])
            dC_dK_list.append(np.argmax(np.sum(weighted_marix, axis=1))+1)
        data["dC_dK_class"] = dC_dK_list
    if flag:
        data.to_json(conf.get("data", "analysis_path"))
        print("cal most likely class by dC_dK finished")


if __name__ == '__main__':
    K = 10
    get_analysis_data(K)
    get_train_valid_result()
    cal_dC_dK(K)
    cal_most_likely_class_by_ovo()
    cal_most_likely_class_by_dcdk()



