import torch
from config import LLAMA2_CONFIG_7B
from calssification_finetune.training import model
from calssification_finetune.data_per import train_data
from tokenizer import tokenizer


model.load_state_dict(torch.load("calssification_finetune/review_classifier.pth"))
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    input_ids = input_ids[:min(max_length, LLAMA2_CONFIG_7B["context_length"])]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    label=torch.softmax(logits,dim=-1)
    predicted_label = torch.argmax(label, dim=-1).item()
    
    return list(train_data.classes.keys())[predicted_label]




def classifer(text):
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    return classify_review(
        text, model, tokenizer, device, max_length=train_data.max_length
    )
