import torch
from torch import nn

class BaseSentenceEncoder(nn.Module):
    pass

class SentenceClassifier(nn.Module):
    pass

class AWESentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, vocab) -> None:
        super(AWESentenceEncoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim, requires_grad = False)
    
    def forward(self, input):
        embeddings = self.embed(input)
        return embeddings.mean(1)

    
class UnidirectionalLSTMSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(AWESentenceEncoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim, requires_grad = False)
        self.rnn = nn.LSTMCell(embedding_dim, hidden_dim)


    def forward(self, input):
        batch_size = input.size(0)
        num_tokens = input.size(1)

        input_ = self.embed(input)
        hx = input_.new_zeros(batch_size, self.rnn.hidden_size)
        cx = input_.new_zeros(batch_size, self.rnn.hidden_size)

        for i in range(num_tokens):
            hx, cx = self.rnn(input_[i], (hx, cx))

        #Return the last hidden cell state
        return hx
        

class SimpleBiLSTMSentenceEncoder(BaseSentenceEncoder):
    
    pass

class BiLSTMSentenceEncoder(BaseSentenceEncoder):
    pass