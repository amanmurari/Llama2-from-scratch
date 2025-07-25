import torch
from torch.utils.data import Dataset,DataLoader
from ..tokenizer import tokenizer
from .. config import LLAMA2_CONFIG_7B

class TextData(Dataset):
    def __init__(self,txt, tokenizer,max_length,stride):
        self.input_ids=[]
        self.output_ids=[]
        token_ids=tokenizer.encode(txt)
        for i in range(0,len(token_ids)-max_length,stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.output_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

        
               
def create_dataloader(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    

    # Create dataset
    dataset = TextData(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
with open("data/data.txt",'r') as f:
    text_data= f.read()
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=LLAMA2_CONFIG_7B["context_length"],
    stride=LLAMA2_CONFIG_7B["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=LLAMA2_CONFIG_7B["context_length"],
    stride=LLAMA2_CONFIG_7B["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)