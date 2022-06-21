import requests
import torch

from ogb.nodeproppred import NodePropPredDataset
from typing import Tuple
from tqdm import tqdm


def rand_edgelist(num_edges=10, num_nodes=100):
    return torch.randint(num_nodes, (2, num_edges))


def sparse_adjacency(edgelist, num_nodes):
    adjacency = torch.sparse_coo_tensor(edgelist,
                                        torch.ones(edgelist.shape[1]),
                                        size=(num_nodes, num_nodes))
    return adjacency


def sparse_identity(length):
    coos = torch.stack((torch.arange(0, length), torch.arange(0, length)))
    identity = torch.sparse_coo_tensor(coos, torch.ones(length))
    return identity


def download_title_abstract_data(data_path: str):

    fname = "titleabs.tsv.gz"
    url_file = f"https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/{fname}"

    print(f"Downloading {url_file}...")
    r = requests.get(url_file, stream=True)

    with open(f"{data_path}{fname}", 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length/1024) + 1, desc="Downloading title-abs data..."):
            if chunk:
                f.write(chunk)
                f.flush()


def _prepare_dataset(dataset):

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, labels = dataset[0]

    num_nodes = graph["num_nodes"]

    features = torch.tensor(graph["node_feat"])
    labels = torch.LongTensor(labels).squeeze()
    train_idx = torch.LongTensor(train_idx)
    valid_idx = torch.LongTensor(valid_idx)
    test_idx = torch.LongTensor(test_idx)

    # build symmetric adjacency
    adj = sparse_adjacency(graph["edge_index"], num_nodes=num_nodes)
    adj = adj + torch.transpose(adj, 0, 1)
    adj = adj + sparse_identity(num_nodes)

    return adj, features, labels, train_idx, valid_idx, test_idx


def OGBNArxiv(data_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:

    d_name = "ogbn-arxiv"
    dataset = NodePropPredDataset(name=d_name, root=data_path)

    download_title_abstract_data(data_path)

    return _prepare_dataset(dataset)
