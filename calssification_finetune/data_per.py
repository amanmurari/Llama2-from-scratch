
import pandas as pd
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader


tokenizer= tiktoken.get_encoding("gpt2")
class ClassifyData(Dataset):
    def __init__(self,csv,tokenizer,max_length=None, pad_token_id=50256):
        self.df = self._df_process(csv)

        self.input_ids=[tokenizer.encode(text) for text in self.df["text"]]
        self.output_ids=[ids for ids in self.df['labeli']]
        if max_length is None:
           max_length= max([len(i) for i in self.input_ids])

        self.max_length=max_length
        self.input_ids=[
            inputs[:max_length]
            for inputs in self.input_ids
        ]
        self.encoded= [
            inputs+[pad_token_id]*(max_length-len(inputs))
            for inputs in self.input_ids
        ]


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return (
           torch.tensor(self.encoded[index],dtype=torch.long),
            torch.tensor(self.output_ids[index],dtype=torch.long)
        )
        
    def _df_process(self,csv) -> pd.DataFrame:
        df = pd.read_csv(csv,names=['a','company','label','text'])
        df=df[~df.text.isna()]
        df=df[~df.label.isna()]
        uni= df.label.unique()
        d={uni[i]: i for i in range(len(uni))}
        self.classes=d
        df["labeli"]=df["label"].map(d)
        return df


train_data= ClassifyData("data/twitter_training.csv",tokenizer)       
train_loader= DataLoader(train_data,batch_size=10,shuffle=True)


val_data= ClassifyData("data/twitter_validation.csv",tokenizer,train_data.max_length)       
val_loader= DataLoader(val_data,batch_size=10,shuffle=True)
