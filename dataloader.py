import torch
from torch.utils.data import DataLoader, Dataset

class PhraseDataset(Dataset):
    def __init__(self,df,pad_sequences):
        super().__init__()
        self.df = df
        self.pad_sequences = pad_sequences
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        if 'Sentiment' in self.df.columns:
            label = self.df['Sentiment'].values[idx]
            item = self.pad_sequences[idx]
            return item,label
        else:
            item = self.pad_sequences[idx]
            return item