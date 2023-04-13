
from data import get_minibatch, prepare_minibatch
import torch

def evaluate_minibatch(model, criterion, data, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=64, device = "cpu"):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout
    total_loss = 0
    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x_premise_packed, x_hypothesis_packed, targets = prep_fn(mb, model.vocab, device)
        with torch.no_grad():
            logits = model(x_premise_packed, x_hypothesis_packed)
            B = targets.size(0)
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            total_loss = total_loss + loss

        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total), total_loss