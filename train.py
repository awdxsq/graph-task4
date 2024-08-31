import torch
from torch.optim import Adam
from load_dataset import load_dataset
from preprocess_data import preprocess_data
from models import TransE,RotatE,ConvE

def train(dataset, model, optimizer):
    model.train()
    for data in dataset:
        h, r, t = data
        optimizer.zero_grad()
        loss = model(h, r, t)
        loss.backward()
        optimizer.step()

# Main training loop
def main():
    dataset = load_dataset('FB15k-237')
    dataset = preprocess_data(dataset)
    model = TransE(dataset.num_entities, dataset.num_relations, 100)
    optimizer = Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        train(dataset, model, optimizer)

if __name__=="__main__":
              main()