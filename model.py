import torch
from torch import nn 

class SentimentRNN(nn.Module):
    
    def __init__(self,corpus_size,output_size,embedd_dim,hidden_dim,n_layers):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(corpus_size,embedd_dim)
        self.lstm = nn.LSTM(embedd_dim, hidden_dim,n_layers,dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim,output_size)
        self.act = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds,hidden)
        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.act(out)
        out = out.view(batch_size,-1)
        out = out[:,-5:]
        return out, hidden
    def init_hidden(self,batch_size):
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden