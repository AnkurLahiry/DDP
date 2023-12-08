import os
import argparse
import random
from mpi4py import MPI
from optuna.trial import TrialState
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time

# Ignore a specific type of warning
warnings.filterwarnings("ignore")

#warnings.filterwarnings("ignore", category=dgl.DGLWarning)

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[: mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

def download_files(comm, rank):
    dataset = None
    if rank == 0:
        dataset = DglNodePropPredDataset("ogbn-arxiv")
    dataset = comm.bcast(dataset, root=0)
    graph, node_labels = dataset[0]
    #Add reverse edges since ogbn-arxiv is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata["label"] = node_labels[:, 0]

    node_features = graph.ndata["feat"]
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()

    idx_split = dataset.get_idx_split()
    train_nids = idx_split["train"]
    valid_nids = idx_split["valid"]
    test_nids = idx_split["test"]  # Test node IDs, not used in the tutorial though.
    print(f'{rank}: {graph}')
    return {"graph": graph, "labels":node_labels, "node_feats": node_features, "train_nids": train_nids, "valid_nids": valid_nids, "test_nids": test_nids}

def get_dataloader(information, device_id=None):
    graph = information["graph"]
    node_feats = information["node_feats"]
    train_nids = information["train_nids"]
    valid_nids = information["valid_nids"]
    test_nids = information["test_nids"]
    device = None
    if device_id is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_id}")
    sampler = dgl.dataloading.NeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        device=device,
        use_ddp=True,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        device=device,
        use_ddp=True,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader, test_dataloader

def task(train_dataloader, valid_dataloader, test_dataloader, graph_information, device_id=None):
    node_features = graph_information["node_feats"]
    num_features = node_features.shape[1]
    labels = graph_information["labels"]
    num_classes = (labels.max() + 1).item()
    device = None
    if device_id is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_id}")

    model = Model(num_features, 128, num_classes).to(device)
    if device == torch.device("cpu"):
        #model = Model(num_features, 128, num_classes).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = Model(num_features, 128, num_classes).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0
    best_model_path = "./model.pt"

    start_time = time.time()

    train_loss = []
    validation_loss = []

    for epoch in range(1000):
        model.train()

        t_loss = []
        v_loss = []
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )

                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )
                t_loss.append(loss.item())
        

        
        predictions = []
        labels = []
        with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata["feat"]
                labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                pred = model(mfgs, inputs)
                predictions.append(
                    model(mfgs, inputs).argmax(1).cpu().numpy()
                )
                l = mfgs[-1].dstdata["label"]
                print(type(pred))
                print(type(labels))
                loss = F.cross_entropy(pred, l)
                v_loss.append(loss.item())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
        train_loss.append(np.average(t_loss))
        validation_loss.append(np.average(v_loss))

    end_time = time.time()

#    plt.figure(figsize=(8, 6))
#    plt.plot(train_loss, label='Training Loss')
#    plt.plot(validation_loss, label='Validation Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.title('Training and Validation Loss')
#    plt.legend()
#
#    # Save the figure to a file
#    plt.savefig('training_validation_loss_plot.png')  # Change the filename and extension as needed
#    plt.show()

    return abs(end_time - start_time)
        

def parser():
    parser = argparse.ArgumentParser(description='Arguments for SSGNN')
    parser.add_argument('-g','--gpus',type=int,metavar='<count>',
                      help='The number of GPUs per node')
    parser.add_argument('-n','--num_proc',type=int,metavar='<count>',
                      help='The total number of processes')
    parser.add_argument('-r','--rank',type=int,metavar='<number>',
                      help='The process rank')
    parser.add_argument('-a','--address',metavar='ip:port',default='tcp://localhost:8888',
                      help='Address of the root process')
    parser.add_argument('-s','--seed',type=int,metavar='<number>',default=42,
                      help='An integer to \"fix\" randomness')
    parser.add_argument('-w','--workdir',metavar='<work directory>',
                      help='Base directory for working files')
    args = vars(parser.parse_args())
    return args

def set_random_seeds(random_seed = 0):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def init_dist(address, rank, num_proc):
    dist.init_process_group(backend='gloo', init_method=address, rank=rank, world_size=num_proc)
    

def get_world_size():
    return dist.get_world_size()

def get_current_rank():
    return dist.get_rank()

def build_graph(comm, rank, size):
     
    chunk_size = len(self.df) // size
    remaining = len(self.df) % size
        
    if rank < remaining:
        chunk_size += 1
        start = rank * chunk_size
    else:
        start = rank * chunk_size + remaining

    end = start + chunk_size

    subgraph = nx.DiGraph()

    edges_df = pd.read_csv('facebook/edges.csv', nrows=end_line - start_line)
    
    for index, row in edges_df.iterrows():
        source = row["id_1"]
        destination = row["id_2"]

        if source not in subgraph:
            subgraph.add_node(source)
        
        if destination not in subgraph:
            subgraph.add_node(destination)

        subgraph.add_edge(source, destination, weight=1/random.randint(100, 999))

    df = pd.read_csv('facebook/features.csv')
    

if __name__ == "__main__":
    arguments = parser()
    set_random_seeds(random_seed = arguments['seed'])
    init_dist(arguments['address'], arguments['rank'], arguments['num_proc'])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #df = pd.read_csv('facebook/edges.csv')

    #X = df.drop(["class"], axis=1)
    #y = df["class"]
    
    #build_graph(comm, rank, size)

    print(f'[Rank {rank}] Welcome to the main function')
    
    graph_information = download_files(comm, rank)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(graph_information)

    time = task(train_dataloader, valid_dataloader, test_dataloader, graph_information)

    print(time)

    print(f'[Rank {rank}] Saying goodbye! See you soon')
