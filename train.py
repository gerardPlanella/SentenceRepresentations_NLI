from data import prepare_minibatch, get_minibatch
from evaluation import evaluate_minibatch, test_model_snli
import time
from torch import optim
import torch
from tqdm import tqdm
from models import AWESentenceEncoder, LSTMEncoder, BiLSTMEncoder
from datetime import datetime as d

#TODO: Add tensorboard support
def train_model(model, dataset, optimizer, criterion ,scheduler, num_epochs, 
                checkpoint_path = "models/",
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate_minibatch,
                batch_size=64, eval_batch_size=None,
                device = "cpu",
                writer = None,
                lr_factor = 5):
    """Train a model."""  
    train_data, dev_data, test_data = dataset.get_data()

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_acc = 0  
    print_every = 1000
    best_eval = 0
    best_iter = 0
    best_model_path = None

    if eval_batch_size is None:
        eval_batch_size = batch_size

    for epoch in tqdm(range(num_epochs)):

        model.train()
        current_loss = 0.
        i = 0
        if writer is not None:
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            
        for batch in batch_fn(train_data, batch_size=batch_size):
            # forward pass
            premise_tup, hypothesis_tup, targets = prep_fn(batch, model.vocab, device)
            logits = model(premise_tup, hypothesis_tup)
            B = targets.size(0)  # later we will use B examples per update
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss.item()
            i = i + 1
            if i%print_every == 0:
                print(f"Batch number {i}")

        
        train_losses.append(current_loss)
        print("Training Loss: " + str(current_loss))
        if writer is not None:
            writer.add_scalar("Training Loss", current_loss, epoch)
        
        _, _, dev_acc, dev_loss = eval_fn(
            model, criterion, dev_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn, device=device)

        val_losses.append(dev_loss.item())
        val_accuracies.append(dev_acc)
        
        print("Validation Loss: " + str(dev_loss.item()))
        if writer is not None:
            writer.add_scalar("Validation Loss", dev_loss, epoch)
        print("Validation Accuracy: " + str(dev_acc))
        if writer is not None:
            writer.add_scalar("Validation Accuracy", dev_acc, epoch)

        if dev_acc > best_eval:
            print("new highscore")
            best_eval = dev_acc
            best_iter = epoch
            best_model_path = createCheckpointPathName(checkpoint_path, model.encoder, dev_acc, model.complex)
            torch.save(model, best_model_path)
            optimizer.param_groups[0]['lr'] /= lr_factor


        if optimizer.param_groups[0]['lr'] < 10**(-5):
            print("Training stopped due to LR limit.")
            break


        scheduler.step()
    
    print("Loading best model to test...")

    test_acc = test_model_snli(best_model_path, test_data, criterion, batch_fn, prep_fn, eval_fn, batch_size, device, writer)
    print("Test Accuracy: " + str(test_acc))

    return train_losses, val_losses, val_accuracies, test_acc
    


 

def createCheckpointPathName(path, model, acc, complex):
    if path[-1] != "/" and path[-1] != "\\":
        path = path + "/"

    cls = model.__class__
    name = cls.__name__
    if cls == BiLSTMEncoder:
        if model.pool_type is not None:
            name = name + "_pooling-" + model.pool_type
    date = d.now().strftime("%Y-%m-%d-%H-%M-%S")

    if not complex:
        return path + name +"_" + str(model.out_dim) +"_"+f"{acc:.2f}" +"_" + date + ".pt"
    else:
        return path + name +"_complex_" +  str(model.out_dim) +"_"+f"{acc:.2f}" +"_" + date + ".pt"
    