from dgl.data import DGLRelationTripleDataset

def preprocess_data(dataset):
    dataset = dataset.map(lambda x: (x[0], x[1], x[2]))
    return dataset