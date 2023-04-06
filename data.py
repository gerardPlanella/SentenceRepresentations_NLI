from typing import Tuple
import numpy as np
from collections import Counter, OrderedDict, defaultdict
import nltk
from tqdm import tqdm
from datasets import load_dataset

class Dataset:
    def __init__(self, dataset_name) -> None:
        self.dataset = load_dataset(dataset_name)      

class NLTKTokenizer():
    def __init__(self):
        self.tokenizer = nltk.word_tokenize

    def encode(self, text):
        return self.tokenizer(text)


class OrderedCounter(Counter, OrderedDict):
  """Counter that remembers the order elements are first seen"""
  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__,
                      OrderedDict(self))
  def __reduce__(self):
    return self.__class__, (OrderedDict(self),)
  

class FeatureVectors:
  def __init__(self) -> None:
    self.features = {}
    self.feature_length = None
    self.vectors = None
  
  def add_feature(self, t, features):
    self.features[t] = features
    if self.feature_length is None:
       self.feature_length = len(features)

  def build(self, vocab):
    self.vectors = [None]*len(vocab.w2i)
    for token in vocab.i2w:
      if token not in self.features:
        #Here we can modify the initialisation of pad and unknown tokens
        #initialized as 0s
        #vectors[v.w2i[token]] = np.random.randn(feature_length)
        if token == "<UNK>":
           self.vectors[vocab.w2i[token]] = [0]*self.feature_length
        if token == "<PAD>":
          self.vectors[vocab.w2i[token]] = [0]*self.feature_length
      else:
        self.vectors[vocab.w2i[token]] = self.features[token]

    self.vectors = np.stack(self.vectors, axis = 0)
    return self.vectors

    

class Vocabulary:
  """A vocabulary, assigns IDs to tokens"""
  
  def __init__(self):
    self.w2i = {}
    self.i2w = []
    self.freqs = OrderedCounter()
    
  def add_token(self, t):
    self.w2i[t] = len(self.w2i)
    self.i2w.append(t)

  def count_token(self, t):
     self.freqs[t] += 1

    
  def build(self, min_freq = 0):
    self.add_token("<UNK>")  # reserve 0 for  unknown words
    self.add_token("<PAD>")  # reserve 1 for  padding 
  
    tok_freq = list(self.freqs.items())
    tok_freq.sort(key=lambda x: x[1], reverse=True)
    for tok, freq in tok_freq:
      if freq >= min_freq:
        self.add_token(tok)

  def __len__(self):
        return len(self.w2i)



def load_embeddings(path = "dataset/glove.840B.300d.txt", tokenizer_cls = NLTKTokenizer, reduced_vocab = False) -> Tuple[Vocabulary, FeatureVectors]:
    vocab = Vocabulary()
    featureVectors = FeatureVectors()
    tokenizer = tokenizer_cls()
    print(f"Loading embeddings from {path}, tokenizing with {tokenizer_cls}.")
    num_lines = sum(1 for line in open(path,'r'))
    idx = 0
    with open(path) as f:
        for line in tqdm(f, total=num_lines):
            idx += 1
            elements = line.split(" ")
            token = elements[0]
            token = tokenizer.encode(token)
            if len(token) > 0:
               token = token[0]
            else:
               continue
            features = list(map(float, elements[1:]))
            vocab.count_token(token)
            featureVectors.add_feature(token, features)
            if reduced_vocab:
              if idx > num_lines / 10:
                break
    vocab.build()
    featureVectors.build(vocab)

    return (vocab, featureVectors)
