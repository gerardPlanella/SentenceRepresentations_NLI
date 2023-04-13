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


encoder = AWESentenceEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
print("Models running on device: ", device)

dataset = CustomDataset(data_percentage=data_percentage)
vocab, featureVectors = load_embeddings(dataset_vocab=dataset.get_vocab(splits=["train"]))
model = SentenceClassifier(len(vocab), embedding_dim, 512, 512, vocab, featureVectors, encoder).to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.1)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.99)
criterion = nn.CrossEntropyLoss()

train_model(model, dataset, optimizer, criterion, scheduler, num_epochs, device = device)

