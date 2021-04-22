import numpy as np
import pandas as pd
from ner import *

def read_tiny_data():
   data = pd.read_csv("data/tiny.ner.train", sep="\t", header=None, names=["word", "label"])
   return data

def test_label_encoding():
   data = read_tiny_data()
   tiny_vocab = label_encoding(data.word.values)
   sample_words = ['5-lipoxygenase', 'NF-kappa', 'which', 'CD28-responsive']
   out = np.array([tiny_vocab[x] for x in sample_words])
   actual = np.array([5, 13, 65, 9])
   assert(np.array_equal(out, actual))
   tiny_labels = label_encoding(data.label.values)
   labels = ['B-DNA', 'B-cell_type', 'B-protein', 'I-DNA', 'I-cell_type', 'I-protein', 'O']
   actual = np.array([0, 1, 2, 3, 4, 5, 6])
   out = np.array([tiny_labels[x] for x in labels])
   assert(np.array_equal(out, actual))

def test_dataset_encoding():
   tiny_data = read_tiny_data()
   tiny_vocab2index = label_encoding(tiny_data["word"].values)
   tiny_label2index = label_encoding(tiny_data["label"].values)
   tiny_data_enc = dataset_encoding(tiny_data, tiny_vocab2index, tiny_label2index)
   actual = np.array([6, 6, 6, 6, 6, 6, 2, 6, 2, 6])
   assert(np.array_equal(tiny_data_enc.iloc[30:40].label.values, actual))
   actual = np.array([17, 53, 31, 25, 44, 41, 32,  0, 11,  1])
   assert(np.array_equal(tiny_data_enc.iloc[30:40].word.values, actual))

def test_Dataset():
   tiny_data = read_tiny_data()
   tiny_vocab2index = label_encoding(tiny_data["word"].values)
   tiny_label2index = label_encoding(tiny_data["label"].values)
   tiny_data_enc = dataset_encoding(tiny_data, tiny_vocab2index, tiny_label2index)
   tiny_ds = NERDataset(tiny_data_enc)
   x, y = tiny_ds[0]
   assert(np.array_equal(x, np.array([11, 30, 26, 18, 13])))
   assert(y == 6)
   assert(len(tiny_ds) == 93)




