import nltk
from models import *
from data import *
from evaluation import evaluate_minibatch
from train import train_model
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import argparse

tokenizers = {
    "nltk":NLTKTokenizer
}

encoders = {
    "awe":AWESentenceEncoder,
    "lstm":LSTMEncoder,
    "bilstm":BiLSTMEncoder
}

encoder_poolings = ["max"]

implemented_datasets = ["snli"]


parser = argparse.ArgumentParser(description='NLI training')


parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--classifier_fc_dim", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--lr_decay", type=float, default=0.99)
parser.add_argument("--encoder_dropout", type=float, default=0.)
parser.add_argument("--encoder_pooling", type=str, default=None)
parser.add_argument("--encoder_lstm_dim", type=int, default=2048)
parser.add_argument("--encoder", type=str, default="bilstm")
parser.add_argument("--data_percentage", type=int, default=50)
parser.add_argument("--reload_dataset", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="snli")
parser.add_argument("--tokenizer", type=str, default="nltk")
parser.add_argument("--dataset_vocab_path", type=str, default="dataset/dataset_vocab.pickle")
parser.add_argument("--vocab_path", type=str, default="dataset/vocab.pickle")
parser.add_argument("--embedding_path", type=str, default="dataset/glove.840B.300d.txt")
parser.add_argument("--checkpoint_path", type=str, default="models/")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=None)


parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

assert params.data_percentage > 0 and params.data_percentage <= 100
assert params.tokenizer in tokenizers
assert params.encoder in encoders
assert params.dataset in implemented_datasets
if params.encoder_pooling is not None:
    assert params.encoder_pooling in encoder_poolings
assert os.path.exists(params.checkpoint_path)

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Models running on device: ", device)

nltk.download('punkt')

dataset = CustomDataset(data_percentage=params.data_percentage, tokenizer_cls=tokenizers[params.tokenizer], dataset_name=params.dataset)
vocab = None
featureVectors = None
if params.reload_dataset:
    dataset_vocab = dataset.get_vocab(splits=["train"], vocab_path=params.dataset_vocab_path, reload=True)
    vocab, featureVectors = load_embeddings(path=params.embedding_path, tokenizer_cls=tokenizers[params.tokenizer], dataset_vocab=dataset_vocab, vocab_path=params.vocab_path, reload=True)
else:
    vocab, featureVectors = load_embeddings(path=params.embedding_path, tokenizer_cls=tokenizers[params.tokenizer], vocab_path=params.vocab_path, reload=False)
vectors = torch.from_numpy(featureVectors.vectors).to(device)

model = SentenceClassifier(len(vocab), params.embedding_dim, params.encoder_lstm_dim, 
                           params.classifier_fc_dim, vocab, vectors, encoders[params.encoder], 
                           encoder_dropout=params.encoder_dropout, encoder_pooling=params.encoder_pooling
                           ).to(device)

optimizer = optim.SGD(model.parameters(), lr = params.lr)
scheduler = lr_scheduler.MultiplicativeLR(optimizer, [params.lr_decay], verbose = True)
criterion = nn.CrossEntropyLoss()

train_model(model, dataset, optimizer, criterion, scheduler, params.num_epochs, 
            device = device, 
            batch_size=params.batch_size, 
            eval_batch_size=params.eval_batch_size,
            checkpoint_path=params.checkpoint_path
            )

