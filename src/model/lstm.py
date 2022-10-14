"""Custom LSTM."""


import torch


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16):
        super(LSTM, self).__init__()
        self.lstm_cell = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size)
        self.W_o = torch.nn.Parameter(torch.rand(output_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.rand(1))
        self.register_buffer("h_t", torch.zeros(batch_size, hidden_size))
        self.register_buffer("c_t", torch.zeros(batch_size, hidden_size))
    
    def forward(self, x, hidden_mask):
        self.h_t, self.c_t = self.h_t * hidden_mask, self.c_t * hidden_mask
        self.h_t, self.c_t = self.lstm_cell(x, (self.h_t, self.c_t))
        return self.h_t @ self.W_o.T + self.b_o
    
    def detach_states(self):
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()
        
    def set_states(self, h_dict, c_dict):
        self.h_t = torch.concat([h for _, h in h_dict.items()])
        self.c_t = torch.concat([c for _, c in c_dict.items()])
