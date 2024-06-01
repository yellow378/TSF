import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Block(nn.Module):
    def __init__(self,d_model,num_layers,c_out,output_steps,device,bias=True,drop_out=0.1):
        super(LSTM_Block,self).__init__()
        self.input_size = d_model
        self.hidden_size = d_model
        self.output_size = c_out
        self.output_steps = output_steps
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        self.fc  = nn.Linear(self.hidden_size,self.output_size)
    def forward(self,x,h=None,c=None):
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        if c is None:
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # LSTM前向传播
        out, _ = self.lstm(x, (h, c))
        out = self.fc(out)
        return out
