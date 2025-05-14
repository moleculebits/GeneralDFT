import torch
import numpy as np
import os
import copy

path = './dicts'

#the values in these tensors were computed from a (very) large subsample of the tensors to streamline standardization 
def standardize_data(molgraph, path_to_dicts='', updateflag=0):
    xmean = torch.tensor([[ 7.6867e-01, -9.4612e+00, -1.4481e-01, -2.7926e-02, -3.2133e-04,
          2.9304e-04, -8.8984e-02, -7.1753e-03,  1.7567e-04, -1.3940e-01,
         -2.7261e-03,  7.4399e-05, -1.5435e+00, -3.2014e-02, -3.4186e-03,
         -8.9341e-01, -1.3224e-01, -1.7227e-02, -1.2590e-03,  1.4873e-03,
         -2.9361e-02,  1.1826e-03,  6.0481e-04, -2.2214e-01, -5.0995e-03,
          1.9501e-04, -1.9133e-01, -3.9428e-02, -2.7271e-03, -3.5484e-01,
         -1.3948e-02, -3.8557e-03, -4.4777e-10, -3.8334e-07]],dtype=torch.float64)

    xstd  = torch.tensor([[ 0.6777, 12.4952,  0.3133,  0.8698,  0.0650,  0.0731,  0.8511,  0.3387,
          0.0191,  1.4894,  0.2005,  0.0705,  6.3571,  0.2858,  0.3249,  4.8021,
          1.2465,  0.3041,  0.1600,  0.0230,  0.5875,  0.0872,  0.0444,  2.4918,
          0.1532,  0.1226,  2.3024,  0.7119,  0.1379,  3.0965,  0.3510,  0.1920,
          0.3639,  0.0966]], dtype=torch.float64)

    emean = torch.tensor([[-1.1342e+00, -7.3861e-02, -1.5781e-03,  8.8172e-03,  4.1220e-04,
          1.5204e-05,  1.2861e+00]], dtype=torch.float64)
    estd = torch.tensor([[0.2318, 0.1974, 0.0332, 0.0121, 0.0042, 0.0004, 0.1909]],dtype=torch.float64)


    ymean = torch.tensor([[-7.3128, -6.8986, -6.3480, -1.3083, -0.6720, -0.1318,  0.2604,  0.6324,
         -0.5745,  0.0728, -0.0287]], dtype=torch.float64)

    ystd  = torch.tensor([[0.7657, 0.7797, 0.6323, 0.8415, 0.7266, 0.7158, 0.7085, 0.6656, 2.5582,
         2.6278, 2.2579]], dtype=torch.float64)
    
    if updateflag == 1:
        graphdic = dict()
        for filename in os.listdir(path_to_dicts):
            if '.pt' in filename:
                graphdic.update(torch.load(os.path.join(path_to_dicts, filename)))

        graphs = list(graphdic.values())

        x = [graph.data.x for (graph, _) in graphs]
        y = [graph.data.y[0:11].reshape(1, -1) for (graph, _) in graphs] #we don't really currently use y
        edge_attr = [graph.data.edge_attr for (graph, _) in graphs]

        x = torch.concat(x, dim=0)
        y = torch.concat(y, dim=0)
        edge_attr = torch.concat(edge_attr, dim=0)

        xmean = x.mean(dim=0, keepdims=True)
        ymean = y.mean(dim=0, keepdims=True)
        emean  = edge_attr.mean(dim=0, keepdims=True)

        xstd  = x.std(dim=0, keepdims=True)
        ystd  = y.std(dim=0, keepdims=True)
        estd  = edge_attr.std(dim=0, keepdims=True)

        print(xmean, xstd, emean, estd, ymean, ystd)

    molgraph.data.x = (molgraph.data.x - xmean)/xstd
    molgraph.data.edge_attr = (molgraph.data.edge_attr - emean)/estd
    molgraph.data.y = (molgraph.data.y.reshape(1, -1) - ymean )/ystd

    return molgraph

def add_graphs(ciddic, proddic, agentdic, reactdic, solvdic):
    for idx, prod in enumerate(ciddic['products']):
        ciddic['products'][idx] = copy.deepcopy(proddic[prod][0])
    for idx, agent in enumerate(ciddic['agents']):
        ciddic['agents'][idx] = copy.deepcopy(agentdic[agent][0])
    for idx, reactant in enumerate(ciddic['reactants']):
        ciddic['reactants'][idx] = copy.deepcopy(reactdic[reactant][0])
    if 'solvents' in ciddic:
        for idx, solv in enumerate(ciddic['solvents']):
            ciddic['solvents'][idx] = copy.deepcopy(solvdic[solv][0])
    return ciddic

def standardize_graphs(graphdic):
    for idx, prod in enumerate(graphdic['products']):
        graphdic['products'][idx] = standardize_data(prod, path, 0)
        if not torch.all(torch.isfinite(prod.data.x)):
            print(prod.cid)
    for idx, agent in enumerate(graphdic['agents']):
        graphdic['agents'][idx] = standardize_data(agent, path, 0)
        if not torch.all(torch.isfinite(agent.data.x)):
            print(agent.cid)
    for idx, reactant in enumerate(graphdic['reactants']):
        graphdic['reactants'][idx] = standardize_data(reactant, path, 0)
        if not torch.all(torch.isfinite(reactant.data.x)):
            print(reactant.cid)
    if 'solvents' in graphdic:
        for idx, solv in enumerate(graphdic['solvents']):
            graphdic['solvents'][idx] = standardize_data(solv, path, 0)
    return graphdic

def add_agent_embeddings(graphdic, gl2embs, smilesdic):
    emb    = []
    labels = []
    for agent in graphdic['agents']:
        emb.append(torch.concat((copy.deepcopy(agent.data.y), gl2embs[agent.cid].reshape(1, -1)), dim=1))
        labels.append(smilesdic[agent.cid])
    labels.sort()
    label = '.'.join(labels)
    emb = torch.stack(emb, dim=0).sum(dim=0, keepdim=True)

    return (graphdic, emb, label)

def add_react_embeddings(graphdic, mol2vecdic, smilesdic):
    emb    = []
    labels = []
    for prod in graphdic['products']:
        emb.append(mol2vecdic[prod.cid])
    prodemb = sum(emb)
    emb = []
    for react in graphdic['reactants']:
        emb.append(mol2vecdic[react.cid])
    reactemb = sum(emb)
    for agent in graphdic['agents']:
        labels.append(smilesdic[agent.cid])
    labels.sort()
    label = '.'.join(labels)
    mol2vecemb = torch.from_numpy(np.concatenate((prodemb, reactemb), axis=0))

    return (graphdic, mol2vecemb, label)