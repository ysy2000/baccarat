import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR
from torch.autograd import Variable

class RNN_model(nn.Module):
    def __init__(self, 
    device, 
    rnn_input_dims, 
    hidden_size, 
    n_layers=1, 
    seq_length = 10,
    dropout_p=0, 
    bidirectional=False, 
    rnn_cell='lstm'):
        super(RNN_model,self).__init__() 
        self.rnn_input_dims = rnn_input_dims
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.seq_length = seq_length
        self.num_directional = 2 if bidirectional else 1
        self.rnn_cell_type = rnn_cell
        self.device = device

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.rnn =  self.rnn_cell(self.rnn_input_dims, self.hidden_size, self.n_layers, dropout=self.dropout_p, bidirectional=self.bidirectional, batch_first=True )

        self.conv_full = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)

        self.fc1 = nn.Linear(self.seq_length * self.hidden_size * self.num_directional,50)
        #self.fc2 = nn.Linear(50,100)
        #self.fc3 = nn.Linear(100,50)
        
        self.fc4 = nn.Linear(50,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers * self.num_directional, x.size(0), self.hidden_size)).to(self.device)  # hidden state

        if self.rnn_cell_type.lower() == 'lstm':
            c_0 = Variable(torch.zeros(self.n_layers * self.num_directional, x.size(0), self.hidden_size)).cuda()  # internal state
            out, hn= self.rnn(x, (h_0, c_0 ))

        elif self.rnn_cell_type.lower() == 'gru':
            out, hn= self.rnn(x, h_0)

        batch_size, seq_len, _ = x.size()

        #출력 전체 펼치기
        out = out.contiguous().view(batch_size,-1)
        x = self.relu(out)
        x = self.fc1(x) # --> torch.Size([2, 100])
        x = self.relu(x)
        #x = self.fc2(x) # --> torch.Size([2, 50])
        #x = self.relu(x)
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)
        print(x)

        return x