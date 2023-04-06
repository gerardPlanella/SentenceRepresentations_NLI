import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




class BaseSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(BaseSentenceEncoder, self).__init__()
        self.out_dim = 0
    
    def forward(self, input):
        pass

class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, classifier_hidden_dim, vocab, encoderType:BaseSentenceEncoder) -> None:
        super(SentenceClassifier, self).__init__()
        
        self.encoder = encoderType(vocab_size, embedding_dim, encoder_hidden_dim, vocab)
        self.linear_1 = nn.Linear(self.encoder.out_dim * 4, classifier_hidden_dim)
        self.linear_2 = nn.Linear(classifier_hidden_dim, 3)
        self.act_fn = nn.Softmax(dim = 1)

    def forward(self, premise, hypothesis):
        u = self.encoder(premise)
        v = self.encoder(hypothesis)

        prod = u * v
        sub = torch.abs(u - v)

        combination = torch.cat((u, v, sub, prod), 1) 
        out1 = self.linear_1(combination)
        out2 = self.linear_2(out1)
        
        return self.act_fn(out2)


        


class AWESentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(AWESentenceEncoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim, requires_grad = False)
        self.out_dim = embedding_dim
    
    def forward(self, input):
        _, lens = pad_packed_sequence(input, batch_first=True)
        assert 0 not in lens
        embeddings = self.embed(input) #batch_size, n_words, embedding_dim
        summed = embeddings.sum(1) #batch_size, embedding_dim
        avg = torch.div(summed.transpose(0, -1), torch.IntTensor(lens)) #embedding_dim, batch_size
        return avg.transpose(0, -1)#batch_size, embedding_dim

    
class UnidirectionalLSTMSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(UnidirectionalLSTMSentenceEncoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim, requires_grad = False)
        self.rnn = nn.LSTMCell(embedding_dim, hidden_dim)
        self.out_dim = hidden_dim


    def forward(self, input):
        _, lens = pad_packed_sequence(input, batch_first=True)
        assert 0 not in lens
        batch_size = input.size(0)
        num_tokens = input.size(1)

        input_ = self.embed(input)
        hx = input_.new_zeros(batch_size, self.rnn.hidden_size)
        cx = input_.new_zeros(batch_size, self.rnn.hidden_size)

        outputs = []

        for i in range(num_tokens):
            hx, cx = self.rnn(input_[i], (hx, cx))
            outputs.append(hx)

        #Return the last hidden cell state
        if batch_size == 1:
            return hx
        else:
            indices = torch.IntTensor(lens) - 1
            outputs = torch.cat(outputs).transpose(1, 0) #batch_size, num_tokens, hidden_size
            return outputs[:, indices, :] #batch_size, hidden_size


            
        

class SimpleBiLSTMSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(SimpleBiLSTMSentenceEncoder, self).__init__()
        self.vocab = vocab
        self.fwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.bwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.out_dim = 2*hidden_dim


    def forward(self, input):
        _, lens = pad_packed_sequence(input, batch_first=True)
        assert 0 not in lens
        batch_size = input.size(0)
        num_tokens = input.size(1)

        input_ = self.embed(input)
        h_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_fwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        h_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
        c_bwd = input_.new_zeros(batch_size, self.rnn.hidden_size)
    
        outputs_fwd = []

        for i in range(num_tokens):
            h_fwd, c_fwd = self.rnn(input_[i], (h_fwd, c_fwd))
            outputs_fwd.append(h_fwd)

        for i in reversed(range(num_tokens)):
            h_bwd, c_bwd = self.rnn(input_[i], (h_bwd, c_bwd))



        if batch_size == 1:
            #Return the concatenation between the last hidden stated of the forward and backward LSTMs
            return torch.cat((h_fwd, h_bwd), 1)#batch_size, hidden_size * 2
        else:
            #Last hidden state of forward pass
            indices_fwd = torch.IntTensor(lens) - 1
            outputs_fwd = torch.cat(outputs_fwd).transpose(1, 0) #batch_size, num_tokens, hidden_size
            outputs_fwd = outputs_fwd[:, indices_fwd, :] #batch_size, hidden_size

            #Last hidden state of backward pass are the same, padding is in the beginning of the backward pass
            return torch.cat((outputs_fwd, h_bwd), 1)#batch_size, hidden_size * 2
            




class BiLSTMSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab) -> None:
        super(SimpleBiLSTMSentenceEncoder, self).__init__()
        self.vocab = vocab
        self.fwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.bwdLSTM = UnidirectionalLSTMSentenceEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab)
        self.out_dim = 2*hidden_dim
    def forward(self, input):
        _, lens = pad_packed_sequence(input, batch_first=True)
        assert 0 not in lens

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
        if batch_size == 1:
            #Max Pooling accross each dimension for all hidden states
            return torch.max(h_stack, 0)#batch_size, hidden_size * 2
        else:
            #TODO: Handle Padded tokens
            




