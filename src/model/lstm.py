"""Custom LSTM."""


import torch


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16):
        super(LSTM, self).__init__()
        self.lstm_cell_1 = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size)
        self.W_o = torch.nn.Parameter(torch.rand(output_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.rand(1))
        self.register_buffer("h_t_1", torch.zeros(batch_size, hidden_size))
        self.register_buffer("c_t_1", torch.zeros(batch_size, hidden_size))
        self._init_parameters()
    
    def _init_parameters(self):
        for name, param in self.lstm_cell_1.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(param, std=0.1)
            else:
                torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('tanh', param))

    def forward(self, x):
        self.h_t_1, self.c_t_1 = self.lstm_cell_1(x, (self.h_t_1, self.c_t_1))
        return self.h_t_1 @ self.W_o.T + self.b_o
    
    def detach_states(self):
        self.h_t_1 = self.h_t_1.detach()
        self.c_t_1 = self.c_t_1.detach()
        self.h_t_1.fill_(0)
        self.c_t_1.fill_(0)
        
    def set_states(self, h_dict, c_dict):
        self.h_t_1 = torch.concat([h for _, h in h_dict[1].items()])
        self.c_t_1 = torch.concat([c for _, c in c_dict[1].items()])


class DeepLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16):
        super(DeepLSTM, self).__init__()
        self.lstm_cell_1 = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size[0])
        self.lstm_cell_2 = torch.nn.LSTMCell(
            input_size=hidden_size[0],
            hidden_size=hidden_size[1])
        self.W_o = torch.nn.Parameter(torch.rand(output_size, hidden_size[0]+hidden_size[1])*0.01)
        self.b_o = torch.nn.Parameter(torch.rand(1)*0.01)
        self.register_buffer("h_t_1", torch.zeros(batch_size, hidden_size[0]))
        self.register_buffer("c_t_1", torch.zeros(batch_size, hidden_size[0]))
        self.register_buffer("h_t_2", torch.zeros(batch_size, hidden_size[1]))
        self.register_buffer("c_t_2", torch.zeros(batch_size, hidden_size[1]))
        self._init_parameters()
    
    def _init_parameters(self):
        for name, param in self.lstm_cell_1.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(param, std=0.1)
            else:
                torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('tanh', param))

        for name, param in self.lstm_cell_2.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(param, std=0.1)
            else:
                torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('tanh', param))
    
    def forward(self, x):
        self.h_t_1, self.c_t_1 = self.lstm_cell_1(x, (self.h_t_1, self.c_t_1))
        self.h_t_2, self.c_t_2 = self.lstm_cell_2(self.h_t_1, (self.h_t_2, self.c_t_2))
        self.h_t_2 + self.h_t_1  # skip connection
        self.h_t_12 = torch.concat([self.h_t_1, self.h_t_2], dim=1)  # layer-wise outputs concat
        return self.h_t_12 @ self.W_o.T + self.b_o

    def detach_states(self):
        self.h_t_1 = self.h_t_1.detach()
        self.c_t_1 = self.c_t_1.detach()
        self.h_t_2 = self.h_t_2.detach()
        self.c_t_2 = self.c_t_2.detach()
        self.h_t_1.fill_(0)
        self.c_t_1.fill_(0)
        self.h_t_2.fill_(0)
        self.c_t_2.fill_(0)
    
    def set_states(self, h_dict, c_dict):
        # make lists to sort
        h_1 = [(i, h) for i, h in h_dict[1].items()]
        h_2 = [(i, h) for i, h in h_dict[2].items()]
        c_1 = [(i, h) for i, h in c_dict[1].items()]
        c_2 = [(i, h) for i, h in c_dict[2].items()]
        
        # sort
        h_1 = sorted(h_1, key=lambda x: x[0])
        h_2 = sorted(h_2, key=lambda x: x[0])
        c_1 = sorted(c_1, key=lambda x: x[0])
        c_2 = sorted(c_2, key=lambda x: x[0])
        
        self.h_t_1 = torch.concat([h for _, h in h_1])
        self.c_t_1 = torch.concat([c for _, c in c_1])
        self.h_t_2 = torch.concat([h for _, h in h_2])
        self.c_t_2 = torch.concat([c for _, c in c_2])
