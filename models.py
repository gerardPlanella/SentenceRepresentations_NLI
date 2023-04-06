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
        super(UnidirectionalLSTMSentenceEncoder, self).__init__()
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(SimpleBiLSTMSentenceEncoder, self).__init__()
        self.vocab = vocab
        self.fwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.bwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        


    def forward(self, input):
        batch_size = input.size(0)
        num_tokens = input.size(1)

        input_ = self.embed(input)
        h_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        h_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)

        for i in range(num_tokens):
            h_fwd, c_fwd = self.rnn(input_[i], (h_fwd, c_fwd))

        for i in reversed(range(num_tokens)):
            h_bwd, c_bwd = self.rnn(input_[i], (h_bwd, c_bwd))

        #Return the concatenation between the last hidden stated of the forward and backward LSTMs
        return torch.cat((h_fwd, h_bwd), 1)#batch_size, hidden_size * 2

class BiLSTMSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(SimpleBiLSTMSentenceEncoder, self).__init__()
        self.vocab = vocab
        self.fwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.bwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        
    def forward(self, input):
        batch_size = input.size(0)
        num_tokens = input.size(1)

        input_ = self.embed(input)
        h_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        h_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)

        h_fwd_stack = input_.new_zeros(num_tokens, batch_size, self.rnn.hidden_size)
        h_bwd_stack = input_.new_zeros(num_tokens, batch_size, self.rnn.hidden_size)

        for i in range(num_tokens):
            h_fwd, c_fwd = self.rnn(input_[i], (h_fwd, c_fwd))
            h_fwd_stack[i] = h_fwd
        
        
        for i in reversed(range(num_tokens)):
            h_bwd, c_bwd = self.rnn(input_[i], (h_bwd, c_bwd))
            h_bwd_stack[i] = h_bwd


        h_stack = torch.cat((h_fwd_stack, h_bwd_stack), 2) #num_tokens, batch_size, hidden_size * 2
        #Max Pooling accross each dimension for all hidden states
        return torch.max(h_stack, 0)#batch_size, hidden_size * 2



