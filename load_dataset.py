from dgl.data import DGLKGDataset

def load_dataset(name):
    dataset = DGLKGDataset(name)
    return dataset
