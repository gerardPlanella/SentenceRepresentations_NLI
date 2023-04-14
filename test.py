import nltk
from models import *
from data import *
from evaluation import evaluate_minibatch
from train import train_model
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn

nltk.download('punkt')

embedding_dim = 300
hidden_dim = 512
lr = 0.1
num_epochs = 20
data_percentage = 50
encoder_dropout = 0
encoder_pooling = None
encoder_lstm_dim = 2048
classifier_fc_dim = 512
encoder = BiLSTMEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.1
lr_decay = 0.99

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
print("Models running on device: ", device)

dataset = CustomDataset(data_percentage=data_percentage)
dataset_vocab = dataset.get_vocab()
vocab, featureVectors = load_embeddings()
vectors = torch.from_numpy(featureVectors.vectors).to(device)
model = SentenceClassifier(len(vocab), embedding_dim, encoder_lstm_dim, classifier_fc_dim, vocab, vectors, encoder, encoder_dropout=encoder_dropout, encoder_pooling=encoder_pooling).to(device)
optimizer = optim.SGD(model.parameters(), lr = lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, lr_decay)
criterion = nn.CrossEntropyLoss()

train_model(model, dataset, optimizer, criterion, scheduler, num_epochs, device = device)

