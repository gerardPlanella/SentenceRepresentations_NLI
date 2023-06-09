import nltk
from models import *
from data import *
from evaluation import evaluate_minibatch
from train import train_model
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import argparse
from datetime import datetime as d
from torch.utils.tensorboard import SummaryWriter
import sys

def lr_lambda(epoch):
    return 0.99

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
parser.add_argument("--data_percentage", type=int, default=100)
parser.add_argument("--reload_dataset", action='store_true')
parser.add_argument("--dataset", type=str, default="snli")
parser.add_argument("--tokenizer", type=str, default="nltk")
parser.add_argument("--dataset_vocab_path", type=str, default="dataset/dataset_vocab.pickle")
parser.add_argument("--vocab_path", type=str, default="dataset/vocab.pickle")
parser.add_argument("--embedding_path", type=str, default="dataset/glove.840B.300d.txt")
parser.add_argument("--checkpoint_path", type=str, default="models/")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=None)
parser.add_argument("--tensorboard_dir", type = str, default = "runs/")
parser.add_argument("--lr_factor", type = float, default=5)
parser.add_argument("--complex_model", action='store_true')


parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

assert params.num_epochs > 0
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

logdir = params.tensorboard_dir + encoders[params.encoder].__name__ + "_" + str(params.encoder_lstm_dim)

if params.encoder == "bilstm" and params.encoder_pooling is not None:
    logdir += f"_pooling-{params.encoder_pooling}"

if params.complex_model:
    logdir += "_complex"

logdir += f"_{d.now().strftime('%Y-%m-%d-%H-%M-%S')}"

writer = SummaryWriter(logdir)

dataset = CustomDataset(data_percentage=params.data_percentage, tokenizer_cls=tokenizers[params.tokenizer], dataset_name=params.dataset)
vocab = None
featureVectors = None
if params.reload_dataset:
    dataset_vocab = dataset.get_vocab(vocab_path=params.dataset_vocab_path, reload=True)
    vocab, featureVectors = load_embeddings(path=params.embedding_path, tokenizer_cls=tokenizers[params.tokenizer], dataset_vocab=dataset_vocab, vocab_path=params.vocab_path, reload=True, use_tqdm=True)
    sys.exit()
else:
    vocab, featureVectors = load_embeddings(path=params.embedding_path, tokenizer_cls=tokenizers[params.tokenizer], vocab_path=params.vocab_path, reload=False)
vectors = torch.from_numpy(featureVectors.vectors).to(device)

model = SentenceClassifier(len(vocab), params.embedding_dim, params.encoder_lstm_dim, 
                           params.classifier_fc_dim, vocab, vectors, encoders[params.encoder], 
                           encoder_dropout=params.encoder_dropout, encoder_pooling=params.encoder_pooling,
                           complex=params.complex_model
                           ).to(device)

print(model)

optimizer = optim.SGD(model.parameters(), lr = params.lr)
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, verbose = True)
criterion = nn.CrossEntropyLoss()

train_model(model, dataset, optimizer, criterion, scheduler, params.num_epochs, 
            device = device, 
            batch_size=params.batch_size, 
            eval_batch_size=params.eval_batch_size,
            checkpoint_path=params.checkpoint_path,
            writer = writer,
            lr_factor=params.lr_factor
            )

