"""Custom LSTM."""


import torch


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16):
        super(LSTM, self).__init__()
        self.lstm_cell_1 = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size)
        # self.lstm_cell_2 = torch.nn.LSTMCell(
        #     input_size=hidden_size,
        #     hidden_size=hidden_size)
        self.W_o = torch.nn.Parameter(torch.rand(output_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.rand(1))
        self.register_buffer("h_t_1", torch.zeros(batch_size, hidden_size))
        self.register_buffer("c_t_1", torch.zeros(batch_size, hidden_size))
        # self.register_buffer("h_t_2", torch.zeros(batch_size, hidden_size))
        # self.register_buffer("c_t_2", torch.zeros(batch_size, hidden_size))
    
    def forward(self, x, hidden_mask):
        self.h_t_1, self.c_t_1 = self.h_t_1 * hidden_mask, self.c_t_1 * hidden_mask
        self.h_t_1, self.c_t_1 = self.lstm_cell_1(x, (self.h_t_1, self.c_t_1))
        # self.h_t_2, self.c_t_2 = self.h_t_2 * hidden_mask, self.c_t_2 * hidden_mask
        # self.h_t_2, self.c_t_2 = self.lstm_cell_2(self.h_t_1, (self.h_t_2, self.c_t_2))
        return self.h_t_1 @ self.W_o.T + self.b_o
    
    def detach_states(self):
        self.h_t_1 = self.h_t_1.detach()
        self.c_t_1 = self.c_t_1.detach()
        # self.h_t_2 = self.h_t_2.detach()
        # self.c_t_2 = self.c_t_2.detach()
        
    def set_states(self, h_dict, c_dict):
        self.h_t_1 = torch.concat([h for _, h in h_dict.items()])
        self.c_t_1 = torch.concat([c for _, c in c_dict.items()])
