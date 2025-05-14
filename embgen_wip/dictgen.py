#this file creates data structures with role information from output orca files; filtering happens here too
import os
import json
import torch
import orcawrap
from concurrent.futures import ProcessPoolExecutor
#storing individual reactions as dicts, not as memory efficient but faster to load and train

PATH_TO_ORCA  = './orcaouts'              #location of orca outputs to be converted to .pt
CID_FILE      = './dicts/cid_dic'         #with role information and compounds encoded as cid
SMILES_PATH   = './dicts/red_smiles_dic'  #dictionary for converting between cid and smiles
DICTS_PATH    = './ptfiles'               #parsed orca files in .pt
DEST          = './ptout'                 #dest for orca outputs to be converted to .pt

NUM_PROCS = 0 #adjust according to your system

solvs = set(['176','180','6342','241','263','6568','6569','6386','5943','7964','6212','8078','11',
             '8117','3283','8150','8071','6228','679','31275','702','8857','174','753','8900','12679',
             '15355','8058','887','15413','6344','13387','6375','8003','1031','3776','1049','8028',
             '1140','8471','962','7237','7929','7809']) #solvents for blacklisting

msolvs = set(['1049','8471']) #overriding blacklist for these "solvents"

print(len(solvs))
    
def generate_ptfiles(filename):
    ptfile = filename.replace('.out', '.pt')
    if ptfile not in os.listdir(DICTS_PATH):
        filedic = orcawrap.orca_parser(os.path.join(PATH_TO_ORCA, filename), 'cid')
        torch.save(filedic, os.path.join(DEST, ptfile))
        print(filename)
        
if __name__ == '__main__':
    with ProcessPoolExecutor(NUM_PROCS) as exe:
        futures = [exe.submit(generate_ptfiles, filename) for filename in os.listdir(PATH_TO_ORCA)]

    with open(CID_FILE, 'r') as cidlist:
        clist = json.load(cidlist)
        print('number of raw reactions: '+str(len(clist)))

    with open(SMILES_PATH, 'r') as smilesdic:
        sdic = json.load(smilesdic)

    molgraphs = dict()
    for filename in os.listdir(DICTS_PATH):
        if '.pt' in filename:
            molfile = os.path.join(DICTS_PATH, filename)
            molgraphs.update(torch.load(molfile))

    solvs = set(['176','180','6342','241','263','6568','6569','6386','5943','7964','6212','8078','11','8117','3283','8150','8071','6228','679','31275','702','8857','174','753','8900','12679','15355','8058','887','15413','6344','13387','6375','8003','1031','3776','1049','8028','1140','8471','962','7237','7929','7809'])

    msolvs = set(['1049','8471'])
    
    def pre_filter():
        multreact = 0
        multprod  = 0
        multagent = 0
        noagent   = 0 
        skipset = set()
        countdic = dict()
        for idx, dic in enumerate(clist):
            agentlist = []
            productlist = []
            reactantlist = []
            for role, cid in list(dic.items()):
                cid = str(cid)
                if 'agent' in role:
                    if cid not in solvs or cid in msolvs:
                        agentlist.append(cid)
                    else:
                        del dic[role]
                elif 'product' in role:
                    productlist.append(cid)
                elif 'reactant' in role:
                    reactantlist.append(cid)
            if len(reactantlist) > 3:
                multreact+=1
                skipset.add(idx)
                continue
            if len(productlist) > 1:
                multprod+=1
                skipset.add(idx)
                continue
            if len(agentlist) > 1:
                multagent+=1
                skipset.add(idx)
                continue
            if len(agentlist)== 0:
                noagent+=1
                skipset.add(idx)
                continue

            agentlist.sort()
            cidlab = '.'.join(agentlist)
            countdic[idx] = cidlab

        fewcount=0
        lablist = list(countdic.values())
        for key, val in countdic.items():
            count = lablist.count(val)
            if count < 15:
                fewcount+=1
                skipset.add(key)
        print(f'rxns with too many reactants:{multreact}\n')
        print(f'rxns with too many products:{multprod}\n')
        print(f'multiagent rxns:{multagent}\n')
        print(f'noagent:{noagent}\n')
        print(f'fewcount:{fewcount}\n')
        return skipset

    skipset = pre_filter()
    rlist      = []
    agentdic   = dict()
    solventdic = dict()
    reactdic   = dict()
    proddic    = dict()
    
    for idx, dic in enumerate(clist):
        if idx in skipset:
            continue
        rdic    = dict()
        reacts  = []
        prods   = []
        agents  = []
        solvs   = []
        
        reactstr = ''
        prodstr  = ''
        agentstr = ''
        solvstr  = ''
        inlist   = 1 

        for role, cid in dic.items():
            cid = str(cid)
            if cid not in molgraphs:
                inlist = 0
                #print(cid)
                break
            elif 'reactant' in role:
                reacts.append(cid)
                reactdic[cid] = (molgraphs[cid], sdic[cid])
                if reactstr == '':
                    reactstr += sdic[cid]
                else:
                    reactstr += '.' + sdic[cid]

            elif 'product' in role:
                prods.append(cid)
                proddic[cid] = (molgraphs[cid], sdic[cid])
                if prodstr == '':
                    prodstr += sdic[cid]
                else:
                    prodstr += '.' + sdic[cid]

            elif 'agent' in role:
                agents.append(cid)
                agentdic[cid] = (molgraphs[cid], sdic[cid])        
                if agentstr == '':
                    agentstr += sdic[cid]   # kept for future use
                else:
                    agentstr += '.' + sdic[cid]

            elif 'solvent' in role:
                solvs.append(cid)
                solventdic[cid] = (molgraphs[cid], sdic[cid])
                if solvstr == '':
                    solvstr += sdic[cid]   # kept for future use
                else:
                    solvstr += '.' + sdic[cid] 

        #all reactions have at least one agent, but solvents were not necessarily reported
        if inlist == 0 or agentstr == '':
            continue  
        else:
            smarts = reactstr + '>' + '>' + prodstr
        
        rdic['reactants'] = reacts
        rdic['products']  = prods
        rdic['agents']    = agents
        rdic['solvents']  = solvs
        rdic['smarts']    = smarts

        rlist.append(rdic.copy())

    torch.save(rlist,      'role_list.pt')
    torch.save(agentdic,   'agent_graph_dic.pt')
    torch.save(solventdic, 'solvent_graph_dic.pt')
    torch.save(proddic,    'product_graph_dic.pt')
    torch.save(reactdic,   'reactant_graph_dic.pt')
