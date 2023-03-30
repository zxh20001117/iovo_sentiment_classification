import random
import time

import numpy as np
import pandas as pd
import torch
from configparser import ConfigParser
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')


params = {
    "seed_val": conf.getint("train", "seed_val"),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "epochs": conf.getint("train", "epochs"),
    "batch_size": conf.getint("train", "batch_size"),
    "seq_length": conf.getint("train", "seq_length"),
    "lr": conf.getfloat("train", "lr"),
    "eps": conf.getfloat("train", "eps"),
    "pretrained_model": conf.get("train", "pretrained_model"),
    "test_size": conf.getfloat("train", "test_size"),
    "random_state": conf.getint("train", "random_state"),
    "add_special_tokens": conf.getboolean("train", "add_special_tokens"),
    "return_attention_mask": conf.getboolean("train", "return_attention_mask"),
    "pad_to_max_length": conf.getboolean("train", "pad_to_max_length"),
    "do_lower_case": conf.getboolean("train", "do_lower_case"),
    "return_tensors": conf.get("train", "return_tensors"),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(conf.getint("train", "seed_val"))
np.random.seed(conf.getint("train", "seed_val"))
torch.manual_seed(conf.getint("train", "seed_val"))
torch.cuda.manual_seed_all(conf.getint("train", "seed_val"))


def get_data(path):
    startTime = time.time()
    df = pd.read_excel(path).reset_index()
    loadTime = time.time()
    print(f'it takes {int(loadTime - startTime)} s for loading data\n')

    df["sent_token_length"] = df["sentence"].apply(lambda x: len(x.split()))
    tokenizer = BertTokenizer.from_pretrained(conf.get("train", "pretrained_model"),
                                          do_lower_case=conf.getboolean("train", "do_lower_case"))
    df["sent_bert_token_length"] = df["sentence"].apply(
        lambda x: len(tokenizer(x, add_special_tokens=conf.getboolean("train", "add_special_tokens"))["input_ids"]))
    token_data = tokenizer.batch_encode_plus(
        df.sentence.values,
        add_special_tokens=conf.getboolean("train", "add_special_tokens"),
        return_attention_mask=conf.getboolean("train", "return_attention_mask"),
        padding='max_length',
        max_length=conf.getint("train", "seq_length"),
        return_tensors=conf.get("train", "return_tensors")
    )
    tokens = pd.DataFrame({'token': list(token_data['input_ids'].numpy())})
    df = pd.concat([df, tokens], axis=1)
    encodingTime = time.time()
    print(f'it takes {int(encodingTime - loadTime)} s for BertTokenizer and Encoding\n')

    return df


def get_iovo_data(data, level_one, level_two):
    iovo_data = data[(data["sentiment"] == level_one) | (data["sentiment"] == level_two)].copy()
    iovo_data['label'] = iovo_data.apply(lambda x: 1 if x['sentiment'] == level_one else 0, axis = 1)
    return iovo_data


def data_split(data):
    data = data.copy().reset_index()
    startTime = time.time()
    X = torch.tensor(np.array([i for i in data['token']]))
    y = torch.from_numpy(data['label'].values).long()

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=conf.getfloat("data_split", "test_size"),
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=conf.getint("data_split", "random_state"))

    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                        test_size=conf.getfloat("data_split", "valid_size"),
                                                        shuffle=True,
                                                        stratify=y_test,
                                                        random_state=conf.getint("data_split", "random_state"))

    # create Tensor datasets
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data = TensorDataset(X_test, y_test)

    # dataloaders
    batch_size = conf.getint("train", "batch_size")

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    endTime = time.time()
    print(f'it takes {int(endTime - startTime)} s for splitting and encapsulating data\n')

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    data = get_data("sentimentData/marked sentences sentiment 2.xlsx")
    print(len(data))
    data = get_iovo_data(data, 1, 2)
