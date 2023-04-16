from typing import Tuple
import numpy as np
from collections import Counter, OrderedDict, defaultdict
import nltk
from tqdm import tqdm
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os


class NLTKTokenizer():
    def __init__(self):
        self.tokenizer = nltk.word_tokenize

    def encode(self, text):
        return self.tokenizer(text)
    

class CustomDataset(Dataset):
    def __init__(self, dataset_name = "snli", tokenizer_cls = NLTKTokenizer, data_percentage:int = 100) -> None:
        assert data_percentage > 0 and data_percentage <= 100
        self.dataset = load_dataset(dataset_name, split = [f"train[:{data_percentage}%]", f"validation[:{data_percentage}%]", f"test[:{data_percentage}%]"])
        self.dataset = {f"{split}": self.dataset[i] for i, split in enumerate(["train", "validation", "test"])}
        self.dataset_name = dataset_name
        self.tokenizer_cls = tokenizer_cls()
        self.preprocessed_dataset = None

    def filter_fn(self, example):
      return example['label'] != -1

    def get_data(self):
        splits = ["train", "validation", "test"]
        if self.preprocessed_dataset is not None:
           return self.preprocessed_dataset
        
        for split in splits:
            self.dataset[split] = self.dataset[split].map(self.preprocess)
            self.dataset[split] = self.dataset[split].filter(lambda x: x["label"] != -1)
        self.preprocessed_dataset = (self.dataset["train"], self.dataset["validation"], self.dataset["test"])
        return self.dataset["train"], self.dataset["validation"], self.dataset["test"]
          
    def preprocess(self, datum):
      if self.dataset_name == "snli":
        datum["premise"] = self.tokenizer_cls.encode(datum["premise"])
        datum["hypothesis"] = self.tokenizer_cls.encode(datum["hypothesis"])
        
        datum["premise"] = [x.lower() for x in datum["premise"]]
        datum["hypothesis"] = [x.lower() for x in datum["hypothesis"]]
        return datum
      else:
        return None
    
    def get_vocab(self, splits=["train", "validation", "test"], vocab_path = "dataset_vocab.pickle", reload = False):
        if self.dataset_name == "snli":

            if os.path.exists(vocab_path) and not reload:
                print("Loading saved Vocabulary from " + vocab_path)
                with open(vocab_path, 'rb') as f:
                    data = pickle.load(f)
                    return data

            train, val, test = self.get_data()
            datasplits = {"train":train, "validation":val, "test":test}
            vocab = {}
            for split in splits:
                data = datasplits[split]
                for datum in data:
                    premise_tokens = set(datum["premise"])
                    hypothesis_tokens = set(datum["hypothesis"])
                    tokens = premise_tokens.union(hypothesis_tokens)
                    for token in tokens:
                        token_stem = self.tokenizer_cls.encode(token)[0].lower()
                        if token_stem not in vocab:
                            vocab[token_stem] = 1

            with open(vocab_path, 'wb') as f:
                #TODO: Save vocab with data_percentage
                print("Saving vocabulary at: " + vocab_path)
                pickle.dump(vocab, f)

            return vocab
        else:
            return None


    
    

class OrderedCounter(Counter, OrderedDict):
  """Counter that remembers the order elements are first seen"""
  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
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



def load_embeddings(path = "dataset/glove.840B.300d.txt", tokenizer_cls = NLTKTokenizer, reduced_vocab = False, dataset_vocab = None, vocab_path = 'vocab.pickle', reload = False, save=True, use_tqdm = False) -> Tuple[Vocabulary, FeatureVectors]:
    if os.path.exists(vocab_path) and not reload:
       print("Loading saved Vocabulary from " + vocab_path)
       return load_vocab(vocab_path)
    
    vocab = Vocabulary()
    featureVectors = FeatureVectors()
    tokenizer = tokenizer_cls()
    print(f"Loading embeddings from {path}, tokenizing with {tokenizer_cls}.")
    num_lines = sum(1 for line in open(path,'r'))
    idx = 0
    with open(path) as f:
        if use_tqdm:
            f = tqdm(f, total=num_lines)
        for line in f:
            idx += 1
            elements = line.split(" ")
            token = elements[0]
            token = tokenizer.encode(token)
            if len(token) > 0:
               token = token[0]
               token = token.lower()
            else:
               continue
            if dataset_vocab is not None:
               #Merge Vocabulary
               if token not in dataset_vocab:
                  continue
            features = list(map(float, elements[1:]))
            vocab.count_token(token)
            featureVectors.add_feature(token, features)
            if reduced_vocab:
              if idx > num_lines / 10:
                break
    vocab.build()
    featureVectors.build(vocab)
    if save:
      print("Saving vocabulary at " + vocab_path)
      save_vocab(vocab, featureVectors, vocab_path)

    return (vocab, featureVectors)

def save_vocab(vocab, featureVectors, path = 'vocab.pickle'):
    data = (vocab, featureVectors)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_vocab(path = 'vocab.pickle'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        vocab, featureVectors = data
        return vocab, featureVectors

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab, device):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([max([len(ex["premise"]), len(ex["hypothesis"])]) for ex in mb])

    x_premise = []
    x_hypothesis = []
    seq_len_prem = []
    seq_len_hyp = []

    for ex in mb:
        seq_len = len(ex["premise"])
        # vocab returns 0 if the word is not there
        padded = pad([vocab.w2i.get(t, 0) for t in ex["premise"]], maxlen)
        x_premise.append(padded)
        seq_len_prem.append(seq_len)

        seq_len = len(ex["hypothesis"])
        # vocab returns 0 if the word is not there
        padded = pad([vocab.w2i.get(t, 0) for t in ex["hypothesis"]], maxlen)
        x_hypothesis.append(padded)
        seq_len_hyp.append(seq_len)



    x_premise = torch.LongTensor(x_premise).to(device)
    seq_len_prem = torch.IntTensor(seq_len_prem)
    seq_len_prem = seq_len_prem.to(device)

    x_hypothesis = torch.LongTensor(x_hypothesis).to(device)
    seq_len_hyp = torch.IntTensor(seq_len_hyp)
    seq_len_hyp = seq_len_hyp.to(device)

    y = [ex["label"] for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return (x_premise, seq_len_prem), (x_hypothesis, seq_len_hyp), y



def get_minibatch(data, batch_size=64, shuffle=True, device="cpu"):
    """Return minibatches, optional shuffling"""

    indices = list(range(len(data)))
    if shuffle:
        print("Shuffling training data")
        random.shuffle(indices)

    batch = []

    # yield minibatches
    for i in indices:
        #TODO: Generate new vocab and delete
        if(data[i]["label"] == -1):
           continue
        batch.append(data[i])

        if len(batch) == batch_size:
            yield batch
            batch = []
            
    # in case there is something left
    if len(batch) > 0:
        yield batch
