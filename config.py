import numpy as np 
import pandas as pd 
from tqdm import tqdm
from collections import Counter
import os
for dirname, _, filenames in os.walk('sisu_tags_quotes.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('data/sisu_tags_quotes.csv')
train = df[:349794]
test  = df[349794:424750]
val   = df[424750:]

def Corpus_Extr(df):
    print('Construct Corpus...')
    corpus = []
    for i in tqdm(range(len(df))):
        corpus.append(df.Phrase[i].lower().split())
    corpus = Counter(np.hstack(corpus))
    corpus = corpus
    corpus2 = sorted(corpus,key=corpus.get,reverse=True)
    print('Convert Corpus to Integers')
    vocab_to_int = {word: idx for idx,word in enumerate(corpus2,1)}
    print('Convert Phrase to Integers')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase_to_int.append([vocab_to_int[word] for word in df.Phrase.values[i].lower().split()])
    return corpus,vocab_to_int,phrase_to_int
corpus,vocab_to_int,phrase_to_int = Corpus_Extr(train)

def Pad_sequences(phrase_to_int,seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length),dtype=int)
    for idx,row in tqdm(enumerate(phrase_to_int),total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences

pad_sequences = Pad_sequences(phrase_to_int,30)