
from data import get_minibatch, prepare_minibatch, CustomDataset
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

def test_model_snli(model_path, test_data, criterion, 
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate_minibatch,
                batch_size=64,
                device = "cpu",
                writer = None):

    model = torch.load(model_path, map_location=device)
    _, _, test_acc, _ = eval_fn(
            model, criterion, test_data, batch_size=batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn, device=device)
    
    if writer is not None:
        writer.add_scalar("Test Accuracy", test_acc)

    return test_acc   

def snli_inference(premise:str, hypothesis:str, model, vocab, device)-> int:
    assert len(premise) > 0 and len(hypothesis) > 0

    #Modify the fields below to your liking
    datum = {
        "premise": premise,
        "hypothesis": hypothesis
    }
    #We tokenize the strings
    

    mb = [datum]

    premise_tup, hypothesis_tup, _ = prepare_minibatch(mb, vocab, device)

    logits = model(premise_tup, hypothesis_tup)

    prediction = logits.argmax(dim=-1)

    return prediction