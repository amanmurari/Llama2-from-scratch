from model import Llama2Model
from config import LLAMA2_CONFIG_7B
from calssification_finetune.data_per import train_loader,val_loader,train_data
import torch
from torch import nn
import tiktoken
import time



tokenizer = tiktoken.get_encoding("gpt2")
device="cuda" if torch.cuda.is_available() else "cpu"
model= Llama2Model(LLAMA2_CONFIG_7B).to(device)
for parms in model.parameters():
    parms.requires_grad=False

model.out_head= nn.Linear(LLAMA2_CONFIG_7B["emb_dim"],len(train_data.classes),device=device,dtype=LLAMA2_CONFIG_7B["dtype"])

for parms in model.trf_blocks[-1].parameters():
    parms.requires_grad=True

for parms in model.final_norm.parameters():
    parms.requires_grad=True


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    model=model.to(device)
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch= input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.inference_mode():
                logits= model(input_batch)[:,-1,:]
            labels= torch.softmax(logits,dim=-1)
            
            labels= torch.argmax(labels,dim=-1)
            correct_predictions+=(labels==target_batch).sum().item()
            
            num_examples += labels.shape[0]
        else:
            break
        
    

    return correct_predictions / num_examples 

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    model= model.to(device)
    logits= model(input_batch)[:,-1,:]
    loss=nn.functional.cross_entropy(nn.functional.softmax(logits,dim=-1),target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
       
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.inference_mode():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter):
   
   
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
        torch.save(model.state_dict(), "calssification_finetune/review_classifier.pth")   
        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen






def trainer():
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 1
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    return train_losses, val_losses, train_accs, val_accs, examples_seen



