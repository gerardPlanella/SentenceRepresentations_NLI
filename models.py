import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class BaseSentenceEncoder(nn.Module):
    def __init__(self) -> None:
        super(BaseSentenceEncoder, self).__init__()
        self.out_dim = 0
    
    def forward(self, input):
        pass

class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, classifier_hidden_dim, vocab, featureVectors, encoderType:BaseSentenceEncoder, encoder_dropout = 0, encoder_pooling = None) -> None:
        super(SentenceClassifier, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        if encoderType == BiLSTMEncoder:
            self.encoder = encoderType(embedding_dim, encoder_hidden_dim, encoder_dropout, encoder_pooling)
        else:
            self.encoder = encoderType(embedding_dim, encoder_hidden_dim, encoder_dropout)

        self.linear_1 = nn.Linear(self.encoder.out_dim * 4, classifier_hidden_dim)
        self.linear_2 = nn.Linear(classifier_hidden_dim, 3)

        with torch.no_grad():
            self.embed.weight.data.copy_(featureVectors)
            self.embed.weight.requires_grad = False

    def forward(self, premise_tup, hypothesis_tup):
        premise_text, premise_len = premise_tup
        hypothesis_text, hypothesis_len = hypothesis_tup
        premise = self.embed(premise_text) #batch_size, n_words, embedding_dim
        hypothesis = self.embed(hypothesis_text)


        u = self.encoder(premise, premise_len)
        v = self.encoder(hypothesis, hypothesis_len)

        prod = u * v
        sub = torch.abs(u - v)

        combination = torch.cat((u, v, sub, prod), 1) 
        out1 = self.linear_1(combination)
        out2 = self.linear_2(out1)
        
        return out2


        


class AWESentenceEncoder(BaseSentenceEncoder):
    def __init__(self, embedding_dim, encoder_dim, dropout ) -> None:
        super(AWESentenceEncoder, self).__init__()
        self.out_dim = embedding_dim
    def forward(self, input, lens):
        summed = input.sum(1) #batch_size, embedding_dim
        avg = torch.div(summed.transpose(0, -1), lens) #embedding_dim, batch_size
        return avg.transpose(0, -1)#batch_size, embedding_dim

class LSTMEncoder(BaseSentenceEncoder):
    def __init__(self, embedding_dim, encoder_dim, dropout) -> None:
        super(LSTMEncoder, self).__init__()
        self.enc_lstm = nn.LSTM(embedding_dim, encoder_dim, 1, bidirectional=False, dropout = dropout)
        self.out_dim = encoder_dim

    def forward(self, sent, sent_len):
        bsize = sent.size(0)
        sent = torch.transpose(sent, 0, 1) #seq_len, batch_size, embedding_dim 

        self.init_lstm = Variable(torch.FloatTensor(1, bsize, self.out_dim).zero_()).to(sent.device)
        # Sort by length (keep idx)

        sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent = torch.index_select(sent, 1, idx_sort)
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, list(sent_len))#.to(original_device)
        sent_output = self.enc_lstm(sent_packed, (self.init_lstm,
                      self.init_lstm))[1][0].squeeze(0)  # batch x 2*nhid

        # Un-sort by length
        idx_unsort = torch.argsort(idx_sort)
        emb = torch.index_select(sent_output, 0, idx_unsort)
        return emb

class BiLSTMEncoder(BaseSentenceEncoder):
    def __init__(self, embedding_dim, encoder_dim, dropout, pooling = None) -> None:
        super(BiLSTMEncoder, self).__init__()
        self.enc_lstm = nn.LSTM(embedding_dim, encoder_dim, 1, bidirectional=True, dropout = dropout)
        self.pool_type = pooling
        self.out_dim = 2*encoder_dim

    
    def forward(self, sent, sent_len):
        sent = torch.transpose(sent, 0, 1) 
        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = torch.sort(sent_len, descending=True)
        idx_unsort = torch.argsort(idx_sort)

        sent = torch.index_select(sent, 1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, list(sent_len_sorted))
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        # batch x seqlen x 2*hid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]
        # batch x seqlen x 2*hid
        sent_output = torch.index_select(sent_output, 0, Variable(idx_unsort))

        if self.pool_type is None:
            fwd_hidden = sent_output[:, 0, :self.enc_lstm.hidden_size]
            bwd_hidden = sent_output[:, -1, self.enc_lstm.hidden_size:]
            sent_output = torch.cat([fwd_hidden, bwd_hidden], dim=-1)
            sent_output = torch.index_select(sent_output, 0, Variable(idx_unsort))
            return sent_output
        
        # Pooling
        if self.pool_type == "max":
            # need to remove zero padding for max pooling
            # list of length batch_size, each element is [seqlen x 2*hid]
            sent_output = [x[:l] for x, l in zip(sent_output, sent_len)]
            emb = [torch.max(x, 0)[0] for x in sent_output]
            emb = torch.stack(emb, 0)
        
        return emb

