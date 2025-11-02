import torch
torch.set_default_dtype(torch.float64)


class DeepNet(torch.nn.Module):
    def __init__(self, activation, n_input, n_hidden=None, n_output=None):
        """
        construct a NN with
        activation: activation function
        n_input: input dimension
        n_hidden: list of hidden layer widths
        n_output: output dim
        """
        super(DeepNet, self).__init__()  # Constructor of the super class torch.nn.Module
        torch.manual_seed(0) # set the seed for reproducibility
        self.dim = n_input
        self.activation = activation
        self.hidden = torch.nn.ModuleList()
        if n_hidden is not None:
            self.L = len(n_hidden)
            self.widths = n_hidden
            self.hidden.append(torch.nn.Linear(n_input, n_hidden[0]))
            torch.nn.init.xavier_normal_(self.hidden[0].weight)
            torch.nn.init.normal_(self.hidden[0].bias)
            for i in range(1, self.L):
                self.hidden.append(torch.nn.Linear(n_hidden[i-1], n_hidden[i]))
                torch.nn.init.xavier_normal_(self.hidden[i].weight)
                torch.nn.init.normal_(self.hidden[i].bias)
        else:
            self.L = 0

        if n_output is not None:
            self.dim_out = n_output
            self.output = torch.nn.Linear(n_hidden[-1], n_output, bias=False)
            torch.nn.init.xavier_normal_(self.output.weight)
        else:
            self.output = None

    def forward(self, x):
        """
        Given input vector x produces the output of the NN
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1) # add a dimension at the end
        for i in range(self.L):
            x = self.hidden[i](x)
            x = self.activation(x)
            
        if self.output is not None:
            x = self.output(x)
        return x