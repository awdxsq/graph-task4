from dgl.nn.pytorch import TransE,RotatE,ConvE

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super(TransE, self).__init__()
        self.transE = TransE(num_entities, num_relations, hidden_dim)

    def forward(self, h, r, t):
        return self.transE(h, r, t)

class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super(RotatE, self).__init__()
        self.rotatE = RotatE(num_entities, num_relations, hidden_dim)

    def forward(self, h, r, t):
        return self.rotatE(h, r, t)

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super(ConvE, self).__init__()
        self.convE = ConvE(num_entities, num_relations, hidden_dim)

    def forward(self, h, r, t):
        return self.convE(h, r, t)