#this file trains gl2vec models and computes compound embeddings
import torch
import os
import argparse
import embgen_wip.tools as tools
from karateclub import gl2vec
from torch_geometric.utils import to_networkx

import pandas as pd
from rdkit.Chem import PandasTools

DICTS_PATH    = './ptfiles' #path to all molgraphs (orca files converted to .pt)

parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='size', type=int, help='specifies size of embedding')
args = parser.parse_args()
 
graphs = dict()
for filename in os.listdir(DICTS_PATH):
    if '.pt' in filename:
        molfile = os.path.join(DICTS_PATH, filename)
        graphs.update(torch.load(molfile))
print(len(graphs))

netgraphdic = dict()
for graph in graphs.values():
    graph = tools.standardize_data(graph)
    edge_attr = graph.data.edge_attr
    x = graph.data.x
    netgraph = to_networkx(graph.data, ["x"], ["edge_attr"])
    netgraphdic[graph.cid] = netgraph


gl2model = gl2vec.GL2Vec(2, args.size, 46)
gl2model.fit(list(netgraphdic.values()))
gl2emb = gl2model.infer(list(netgraphdic.values()))

gl2inputdic = {key: torch.from_numpy(val) for key, val in zip(netgraphdic.keys(), gl2emb)}

torch.save(gl2model, f'gl2model{args.size}wl2.pt', pickle_protocol=4)
torch.save(gl2inputdic, f'gl2iputdic{args.size}.pt', pickle_protocol=4)
