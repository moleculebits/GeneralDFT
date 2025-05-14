#this file takes in the pre-computed gl2vec embeddings, 
#makes mol2vec embeddings and outputs all three reaction embedding types
#if you have issues running it, check your gensim version
import torch
import pandas as pd
import numpy as np
import json

from rdkit import Chem
from rdkit.Chem import PandasTools

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec


product_dic  = torch.load('./dicts/product_graph_dic.pt')
reactant_dic = torch.load('./dicts/reactant_graph_dic.pt')
agent_dic = torch.load('./dicts/agent_graph_dic.pt')
solvent_dic = torch.load('./dicts/solvent_graph_dic.pt')

with open('red_smiles_dic', 'r') as fin:
    sdic = json.load(fin)

gl2inputdic = torch.load('gl2inputdic500.pt') #loads gl2vec embeddings

product_dic = {key: val[1] for key, val in product_dic.items()} #removes unnecessary dic items (we only need smiles)
reactant_dic = {key: val[1] for key, val in reactant_dic.items()}
agent_dic = {key: val[1] for key, val in agent_dic.items()}

mol_dic = dict()
mol_dic.update(product_dic)
mol_dic.update(reactant_dic)
mol_dic.update(agent_dic)

model = word2vec.Word2Vec.load('redmodel500.pkl') #loads mol2vec model

df = pd.DataFrame.from_dict(mol_dic, orient='index', columns=['smiles'])

PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'ROMol', includeFingerprints=True)
df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
df['mol2vec']  = [np.array(DfVec(x).vec) for x in sentences2vec(df['sentence'], model, unseen='UNK')]

molvals = df['mol2vec'].to_list()
molkeys = df.index.to_list()
mol2vecdic = {molkey: molval for molkey, molval in zip(molkeys, molvals)}

mollist = []
mollist.extend(torch.load('molgraphs0.pt'))
mollist.extend(torch.load('molgraphs1.pt'))
mollist.extend(torch.load('molgraphs2.pt'))
mollist.extend(torch.load('molgraphs3.pt'))

mol2input   = []
gl2input    = []
molgl2input = []
for dic in mollist:
    products  = 0
    gl2prod   = 0
    reactants = 0
    gl2react  = 0
    for role, clist in dic.items():
        if 'products' in role:
            for cid in clist:
                products += mol2vecdic[cid]
                gl2prod  += gl2inputdic[cid]
            products = torch.from_numpy(products).reshape(1, -1)
            gl2prod  = gl2prod.reshape(1, -1)
        if 'reactants'in role:
            for cid in clist:
                reactants += mol2vecdic[cid]
                gl2react  += gl2inputdic[cid]
            reactants = torch.from_numpy(reactants).reshape(1, -1)
            gl2react  = gl2react.reshape(1, -1)
        if 'agents' in role:
            agents = [sdic[cid] for cid in clist]
            agents.sort()
            agents = np.array(agents)
        if 'smarts' in role:
            smarts = clist 

    mol2vecemb   = torch.concat((products, reactants), dim=1)
    gl2vecemb    = torch.concat((gl2prod, gl2react), dim=1)
    molgl2vecemb = torch.concat((mol2vecemb, gl2vecemb), dim=1)

    mol2input.append((mol2vecemb, agents, smarts))      #mol2vec embedding, (true) agent, smarts
    gl2input.append((gl2vecemb, agents, smarts))        #gl2vec embedding, (true) agent, smarts
    molgl2input.append((molgl2vecemb, agents, smarts))  #hybrid embedding, (true) agent, smarts

torch.save(mol2input, 'mol2redembs1000.pt')
torch.save(gl2input, 'gl2redembs1000.pt')
torch.save(molgl2input, 'molgl2redembs2000.pt')
