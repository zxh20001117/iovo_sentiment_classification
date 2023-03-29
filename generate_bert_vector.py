import random
import time

import numpy as np
import pandas as pd
import torch
from configparser import ConfigParser
from transformers import BertTokenizer

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
    df = pd.read_excel(path)
    loadTime = time.time()
    print(f'it takes {int(loadTime - startTime)} s for loading data\n')

    df["sent_token_length"] = df["sentence"].apply(lambda x: len(x.split()))
    tokenizer = BertTokenizer.from_pretrained(conf.get("train", "pretrained_model"),
                                          do_lower_case=conf.getboolean("train", "do_lower_case"))
    df["sent_bert_token_length"] = df["sentence"].apply(
        lambda x: len(tokenizer(x, add_special_tokens=conf.getboolean("train", "add_special_tokens"))["input_ids"]))
    encoded_data = tokenizer.batch_encode_plus(
        df.sentence.values,
        add_special_tokens=conf.getboolean("train", "add_special_tokens"),
        return_attention_mask=conf.getboolean("train", "return_attention_mask"),
        padding='max_length',
        max_length=conf.getint("train", "seq_length"),
        return_tensors=conf.get("train", "return_tensors")
    )
    df["input_ids"] = encoded_data["input_ids"]
    df['attention_mask'] = encoded_data['attention_mask']
    encodingTime = time.time()
    print(f'it takes {int(encodingTime - loadTime)} s for BertTokenizer and Encoding\n')

    return df


def get_iovo_data(data, level_one, level_two):
    return data[data["sentiment"] == level_one | data["sentiment"] == level_two]


if __name__ == "__main__":
    data = get_data("sentimentData/marked sentences sentiment 2.xlsx")
    print(len(data))
    print("\ninput_ids:\n", data.head(20)['input_ids'])
    print("\nattention_mask:\n", data.head(20)['attention_mask'])