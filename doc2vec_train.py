from configparser import ConfigParser

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import time

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

startTime = time.time()
data = pd.read_json('data/cutted sentences.json')['sentence']
loadTime = time.time()
print(f'it takes {loadTime - startTime:.1f} s for loading data\n')

tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(data)]
taggingTime = time.time()
print(f'it takes {taggingTime - loadTime:.1f} s for tagging data\n')

vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=conf.getint("doc2vec", "vector_size"),
                alpha=conf.getfloat("doc2vec", "alpha"),
                min_alpha=conf.getfloat("doc2vec", "min_alpha"),
                min_count=conf.getint("doc2vec", "min_count"),
                dm=conf.getint("doc2vec", "dm"),
                window=conf.getint("doc2vec", "window"),
                sample=conf.getfloat("doc2vec", "sample"),
                workers=conf.getint("doc2vec", "workers"),
                negative=conf.getint("doc2vec", "negative"),
                epochs=conf.getint("doc2vec", "epochs")
                )
model.build_vocab(tagged_data)

model.train(tagged_data,
                total_examples=model.corpus_count,
            epochs=model.epochs)
trainingTime = time.time()

print(f'it takes {trainingTime - taggingTime:.1f} s for training data\n')

model.save("./model/doc2vec/d2v.model")
print("Model Saved")
