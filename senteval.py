from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import torch
import os
from models import *
import argparse
import sklearn
from data import NLTKTokenizer, load_embeddings, pad
from torch import nn
from datetime import datetime as d

# path to senteval
senteval_path = '../SentEval'

encoders = {
    "AWESentenceEncoder":AWESentenceEncoder,
    "LSTMEncoder":LSTMEncoder,
    "BiLSTMEncoder":BiLSTMEncoder
}

tokenizers = {
    "nltk": NLTKTokenizer
}



# import SentEval
sys.path.insert(0, senteval_path)
import senteval



def create_dictionary(sentences, tokenizer_cls = NLTKTokenizer):
    tokenizer = tokenizer_cls()
    words = {}
    for s in sentences:
        for word in s:
            processed_word = tokenizer.encode(word)[0].lower()
            words[processed_word] = words.get(word, 0) + 1

    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    return words


def process_batch(mb, vocab, device):
    maxlen = max([len(ex) for ex in mb])
    x = []
    seq_lens = []

    for ex in mb:
        seq_len = len(ex)
        padded = pad([vocab.w2i.get(t, 0) for t in ex], maxlen)
        x.append(padded)
        seq_lens.append(seq_len)

    x = torch.LongTensor(x).to(device)
    seq_lens = torch.IntTensor(seq_lens)
    seq_lens = seq_lens.to(device)

    return x, seq_lens

def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    dataset_vocab = create_dictionary(samples, params.tokenizer_cls)
    params.vocab, featureVectors = load_embeddings(path=params.embedding_path, 
                                                   tokenizer_cls=params.tokenizer_cls, 
                                                   dataset_vocab=dataset_vocab, 
                                                   vocab_path=params.vocab_path, 
                                                   reload=True, save=False)
    vectors = torch.from_numpy(featureVectors.vectors)
    params.embed = nn.Embedding(len(params.vocab), params.embedding_dim)
    with torch.no_grad():
            params.embed.weight.data.copy_(vectors)
            params.embed.weight.requires_grad = False

    params.embed = params.embed.to(params.device)
    return



def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]

    x, seq_lens = process_batch(batch, params.vocab, params.device)

    x_embed = params.embed(x)
    embeddings = params.encoder(x_embed, seq_lens)
    return embeddings.cpu().detach().numpy()

def getEncoderName(model_path):
    assert os.path.isfile(model_path)

    encoder_name = model_path.split("/")[-1].split("_")[0]

    assert encoder_name in encoders 

    if encoder_name == "BiLSTMEncoder":
        if model_path.split("/")[-1].split("_")[1].split("-")[0] == "pooling":
            encoder_name = encoder_name + "_" + model_path.split("/")[-1].split("_")[1]
    
    return encoder_name


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='../SentEval/data/')
    parser.add_argument("--vocab_path", type=str, default="dataset/senteval_vocab.pickle")
    parser.add_argument("--model_path", type=str, default="models/AWESentenceEncoder_300_0.60_2023-04-15-16-47-23.pt")
    parser.add_argument("--embedding_path", type=str, default="dataset/glove.840B.300d.txt")
    parser.add_argument("--kfold", type=int, default=10)
    parser.add_argument("--tokenizer", type=str, default="nltk")
    parser.add_argument("--usepytorch", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--results_path", type=str, default="results/")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder_name = getEncoderName(args.model_path)

    encoder = torch.load(args.model_path, map_location='cpu').encoder.to(device)

    complex_model = args.model_path.find("complex") > -1

    print(encoder)

    params_senteval = {'task_path': args.data_path, 'usepytorch': args.usepytorch, 'kfold': args.kfold,
                       'tokenizer_cls': tokenizers[args.tokenizer], 'vocab_path': args.vocab_path,
                       'embedding_path': args.embedding_path, 'embedding_dim': encoder.embedding_dim,
                       'device':device, 'encoder':encoder}
    
    params_senteval['classifier'] = {'nhid': 0, 'optim': args.optim, 'batch_size': args.batch_size,
                                     'tenacity': 5, 'epoch_size': args.num_epochs}
    

    print("Starting evaluation...")

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['MR', 'CR', 'SUBJ','MPQA',  'SST2', 'TREC', 'MRPC', 'SICKRelatedness',
                      'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)

    results_path = args.results_path 

    if results_path[-1] != "/" and results_path[-1]:
        results_path+="/"

    if complex_model:
        encoder_name += "_complex"
    torch.save(results, args.results_path + encoder_name + f"_sentEval_{d.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt")
    print(results)