import gc
import torch 
from model import SentimentRNN
import numpy as np 
import torch.nn.functional as F 
from torch import nn 
from config import train
from config import vocab_to_int 
from dataloader import PhraseDataset
from dataloader import DataLoader
from config import pad_sequences


vocab_size = len(vocab_to_int)
output_size = 5
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim,n_layers)

net.train()
clip=5
epochs = 200
counter = 0
print_every = 100
lr=0.01

def criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

batch_size=32
losses = []
accs=[]
for e in range(epochs):
    a = np.random.choice(len(train)-1, 1000)
    train_set = PhraseDataset(train.loc[train.index.isin(np.sort(a))],pad_sequences[a])
    train_loader = DataLoader(train_set,batch_size=32,shuffle=True)
    # initialize hidden state
    h = net.init_hidden(32)
    running_loss = 0.0
    running_acc = 0.0
    # batch loop
    for idx,(inputs, labels) in enumerate(train_loader):
        counter += 1
        gc.collect()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        optimizer.zero_grad()
        if inputs.shape[0] != batch_size:
            break
        # get the output from the model
        output, h = net(inputs, h)
        labels=torch.nn.functional.one_hot(labels, num_classes=5)
        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.cpu().detach().numpy()
        running_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if idx%20 == 0:
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format((running_loss/(idx+1))))
            losses.append(float(running_loss/(idx+1)))
            print(f'acc:{running_acc/(idx+1)}')
            accs.append(running_acc/(idx+1))