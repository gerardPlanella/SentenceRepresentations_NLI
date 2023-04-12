
from data import get_minibatch, prepare_minibatch
import torch

def evaluate_minibatch(model, data, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=64):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x_premise_packed, x_hypothesis_packed, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x_premise_packed, x_hypothesis_packed)
            
        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)