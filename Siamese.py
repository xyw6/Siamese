import re
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import keras.backend as K
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from random import sample
from keras.preprocessing.sequence import pad_sequences
import itertools
import numpy as np

def train(X_train, X_val, Y_train, Y_val, embedding, l, n_hidden = 50, grad = 1.25, batch = 64, epoch = 25):
    inputL, inputR = Input(shape=(l,), dtype='int32'), Input(shape=(l,), dtype='int32')
    embedding_layer = Embedding(len(embedding), 300, weights=[embedding], input_length=l, trainable=False)
    encodedL, encodedR = embedding_layer(inputL), embedding_layer(inputR)
    lstm = LSTM(n_hidden)
    outputL, outputR = lstm(encodedL), lstm(encodedR)
    similarity = Lambda(function = lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims = True)), output_shape = lambda x: (x[0][0], 1))([outputL, outputR])
    model = Model([inputL, inputR], [similarity])
    model.compile(loss = 'mse', optimizer = Adadelta(clipnorm = grad), metrics = ['acc', TPR, PPV])
    return model.fit([X_train['L'], X_train['R']], Y_train, batch_size = batch, epochs = epoch, validation_data=([X_val['L'], X_val['R']], Y_val))

def preprocess(filename, vsize = -1):
    df = pd.read_csv(filename)
    q1, q2, is_dupl = df.q1.tolist(), df.q2.tolist(), df.is_dupl.tolist()
    size = len(q1)
    if vsize < 0: vsize = size // 10
    val, tpq, tnq, val_q1, val_q2, val_dupl = set(sample(range(size), vsize)), [], [], [], [], []
    for i in range(size):
        if i in val:
            val_q1.append(q1[i])
            val_q2.append(q2[i])
            val_dupl.append(is_dupl[i])
        elif is_dupl[i] == 0:
            tnq.append((q1[i], q2[i]))
        else:
            tpq.append((q1[i], q2[i]))
    npp, nnp = len(tpq), len(tnq)
    N = min(npp, nnp)
    if npp > nnp:
        tpq = sample(tpq, N)
    else:
        tnq = sample(tnq, N)
    train_q1, train_q2, train_dupl = [], [], []
    for i in range(N):
        train_q1.append(tpq[i][0])
        train_q2.append(tpq[i][1])
        train_dupl.append(1)
        train_q1.append(tnq[i][0])
        train_q2.append(tnq[i][1])
        train_dupl.append(0)
    maxlen, w2id, nw =  0, dict(), 1
    w2v = KeyedVectors.load_word2vec_format('GoogleWord2Vec.bin', binary=True)
    vocab, stops = w2v.vocab, set(stopwords.words('english'))
    for ql in [train_q1, train_q2, val_q1, val_q2]:
        for i in range(len(ql)):
            wl, q2id = q2wl(ql[i]), []
            for w in wl:
                if w in stops and w not in vocab:
                    continue
                if w in w2id:
                    q2id.append(w2id[w])
                else:
                    q2id.append(nw)
                    w2id[w] = nw
                    nw += 1
            ql[i], length = q2id, len(q2id)
            if length > maxlen:
                maxlen = length
    embedding = 1 * np.random.randn(nw, 300)
    for w in w2id:
        if w in vocab:
            embedding[w2id[w]] = w2v.word_vec(w)
    del w2id, w2v, vocab
    X_train, X_val = {'L': pd.Series(train_q1), 'R': pd.Series(train_q2)}, {'L': pd.Series(val_q1), 'R': pd.Series(val_q2)}
    Y_train, Y_val = pd.DataFrame({'is_dupl': train_dupl}), pd.DataFrame({'is_dupl': val_dupl})
    for dataset, side in itertools.product([X_train, X_val], ['L', 'R']):
        dataset[side] = pad_sequences(dataset[side], maxlen = maxlen)
    return X_train, X_val, Y_train.values, Y_val.values, embedding, maxlen

def TPR(true, pred):
    TP = K.sum(K.round(K.clip(true * pred, 0, 1)))
    P = K.sum(K.round(K.clip(true, 0, 1)))
    return TP / (P + K.epsilon())

def PPV(true, pred):
    TP = K.sum(K.round(K.clip(true * pred, 0, 1)))
    PP = K.sum(K.round(K.clip(pred, 0, 1)))
    return TP / (PP + K.epsilon())

def q2wl(q):
    q = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", q.lower())
    q = re.sub(r"\'s", " ", re.sub(r"what's", "what is ", q))
    q = re.sub(r"can't", "cannot ", re.sub(r"\'ve", " have ", q))
    q = re.sub(r"i'm", "i am ", re.sub(r"n't", " not ", q))
    q = re.sub(r"\'d", " would ", re.sub(r"\'re", " are ", q))
    q = re.sub(r",", " ", re.sub(r"\'ll", " will ", q))
    q = re.sub(r"!", " ! ", re.sub(r"\.", " ", q))
    q = re.sub(r"\^", " ^ ", re.sub(r"\/", " ", q))
    q = re.sub(r"\-", " - ", re.sub(r"\+", " + ", q))
    q = re.sub(r"'", " ", re.sub(r"\=", " = ", q))
    q = re.sub(r":", " : ", re.sub(r"(\d+)(k)", r"\g<1>000", q))
    q = re.sub(r" b g ", " bg ", re.sub(r" e g ", " eg ", q))
    q = re.sub(r"\0s", "0", re.sub(r" u s ", " american ", q))
    q = re.sub(r"e - mail", "email", re.sub(r" 9 11 ", "911", q))
    q = re.sub(r"\s{2,}", " ", re.sub(r"j k", "jk", q))
    return q.split()

if __name__ == '__main__':
    print('Preprocessing...')
    X_train, X_val, Y_train, Y_val, embedding, l = preprocess('quora.csv')
    print('Compeleted! Type "test()" to test!')
    def test(): return train(X_train, X_val, Y_train, Y_val, embedding, l)
    