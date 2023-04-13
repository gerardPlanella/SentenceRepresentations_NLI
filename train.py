from data import prepare_minibatch, get_minibatch
from evaluation import evaluate_minibatch
import time
from torch import optim
import torch
from tqdm import tqdm

#TODO: Add tensorboard support
def train_model(model, dataset, optimizer, criterion ,scheduler, num_epochs,
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate_minibatch,
                batch_size=64, eval_batch_size=None,
                device = "cpu"):
    """Train a model."""  
    train_data, dev_data, test_data = dataset.get_data()

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_acc = 0  

    if eval_batch_size is None:
        eval_batch_size = batch_size

    for epoch in tqdm(range(num_epochs)):

        model.train()
        current_loss = 0.
        for batch in batch_fn(train_data, batch_size=batch_size):
            # forward pass
            premise_tup, hypothesis_tup, targets = prep_fn(batch, model.vocab, device)
            logits = model(premise_tup, hypothesis_tup)
            B = targets.size(0)  # later we will use B examples per update
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss

        scheduler.step()
        train_losses.append(current_loss)
        print("Training Loss: " + str(current_loss))
        
        _, _, dev_acc, dev_loss = eval_fn(
            model, criterion, dev_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)

        val_losses.append(dev_loss)
        val_accuracies.append(dev_acc)
        
        print("Validation Loss: " + str(dev_loss))
        print("Validation Accuracy: " + str(dev_acc))

        if optimizer.param_groups[0]['lr'] < 10**(-5):
            print("Training stopped due to LR limit.")
            break
    
    _, _, test_acc, _ = eval_fn(
            model, criterion, test_data, batch_size=batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn, device=device)
    
    print("Test Accuracy: " + str(test_acc))

    return train_losses, val_losses, val_accuracies, test_acc
    