#utility file for parsing to convert raw ORCA outputs to molgraphs used for gl2vec etc.
import cclib
import re
import torch
import numpy as np
from io import StringIO
from torch_geometric.data import Data

class MolData:
    def __init__(self, cid, mult, nelec, atomcoords, scfenergies, atoms, amass, orbitals, dipoles, hirsh, binfo, anboattr, natpop):
        self.cid:         str = cid
        self.mult:        int = mult
        self.nelec:       int = nelec
        self.atomcoords:  np.array = atomcoords
        self.scfenergies: np.array = scfenergies
        self.atoms:       np.array = np.reshape(atoms, (-1,1))
        self.amass:       np.array = np.reshape(amass, (-1,1))
        self.hirsh:       np.array = np.reshape(hirsh, (-1,1))
        self.orbitals:    np.array = np.reshape(orbitals, -1)
        self.dipoles:     np.array = dipoles[1, :]
        self.natpop:      np.array = natpop
        self.bindex:      np.array = None
        self.battr:       np.array = None
        self.anbo:        np.array = None
        self.data:        Data = Data()
        self._binfo:      dict = binfo
        self._anboattr:   list = anboattr
        
        self.setup()

    def setup(self):
        #structuring the atom nbo features
        anbo = np.zeros((len(self.atoms), 31))    #thirty one atomic orbtals per atom (most unoccupied)
        for idx, (anum, orbenergy, occ) in enumerate(self._anboattr):
            anbo[anum][idx%31] = float(orbenergy)*occ

        natcharge = self.natpop[:, 0]
        natcharge = np.reshape(natcharge, (-1,1))

        self.data.x = torch.from_numpy(np.concatenate((self.amass/10, anbo, natcharge, self.hirsh), axis=1)) #total of 34 descriptors

        #structuring the bond features
        self.bindex = np.zeros((2, len(self._binfo)))
        for idx, (atom1, atom2) in enumerate(self._binfo.keys()):
            self.bindex[0][idx] = atom1
            self.bindex[1][idx] = atom2

        self.battr = np.stack(list(self._binfo.values()))

        self.data.edge_index = torch.from_numpy(self.bindex)    #must be in COO format
        self.data.edge_attr  = torch.from_numpy(self.battr)

        #structuring global features
        orbcount = -(-self.nelec//2)       #ceiling division
        globorbs = np.zeros(8)             #selects the 4 highest occupied orbitals and 4 lowest unoccupied orbitals as the CAS    
        for idx in range(orbcount-3, orbcount+5):
            globorbs[idx-(orbcount-3)] = self.orbitals[idx] 
        globfeats = np.concatenate((globorbs, self.dipoles))

        self.data.y = torch.from_numpy(globfeats)

def nbo_reader(nbostring:str, coords, binfo): #for NBO bond info

    nbostring = nbostring.replace('-', ' ', 1)
    nbostring = nbostring.replace('(', ' ', 1)
    nbostring = nbostring.replace(')', ' ', 1)
    nbolist = nbostring.split()

    if 'BD' in nbostring:
        if nbolist[5][-1].isalpha() and nbolist[3][-1].isalpha():
            atom1   = int(nbolist[4])-1
            atom2   = int(nbolist[6])-1
            bval    = float(nbolist[8])*float(nbolist[7])
            btype   = int(nbolist[2])

        elif (not nbolist[5][-1].isalpha()) and nbolist[3][-1].isalpha():
            atom2str = re.findall(r'\d+', nbolist[5])[0]
            atom1    = int(nbolist[4])-1
            atom2    = int(atom2str)-1
            bval     = float(nbolist[7])*float(nbolist[6])
            btype    = int(nbolist[2])
        
        else:
            atom1str = re.findall(r'\d+', nbolist[3])[0]
            atom2str = re.findall(r'\d+', nbolist[4])[0]
            atom1    = int(atom1str)-1
            atom2    = int(atom2str)-1
            bval     = float(nbolist[6])*float(nbolist[5])
            btype    = int(nbolist[2])

        coords1  = np.array(coords[atom1])
        coords2  = np.array(coords[atom2])
        blength = np.sqrt(np.sum((coords1-coords2)**2))    #L2 norm for blengths

        bdic = dict()
        bvec = np.zeros(7)    #number of features = 7
        if nbolist[1] == 'BD':
            bvec[btype-1] = bval
            if btype == 1:
                bvec[6] = blength

        elif nbolist[1] == 'BD*':
            bvec[btype+2] = bval

        bdic[(atom1, atom2)] = bvec

        if (atom1, atom2) in binfo.keys():
            vec = binfo[(atom1, atom2)]
            vec = vec + bvec
            binfo[(atom1, atom2)] = vec
        else:
            binfo.update(bdic.copy())  #stores bond type, indices, legnths, and energy times occupancy


def anbo_reader(anbostring:str, anboattr): #for NBO atom info
    if '-------------------------------------------------------' in anbostring or anbostring[0] == '\n':
        pass
    elif ' Summary of Natural Population Analysis:' in anbostring:
        pass
    else:
        anbolist = anbostring.split()
        if anbolist[1][-1].isalpha():
            anum = int(anbolist[2])-1
            occ      = float(anbolist[6])
            energy   = anbolist[7]

        else:
            anumstr  = re.findall(r'\d+', anbolist[1])[0]
            anum     = int(anumstr)-1
            occ      = float(anbolist[5])
            energy   = anbolist[6]

        anboattr.append((anum, energy, occ))
    
def natpop_reader(instring:str, natpop): #for nat_pop info
    if '-------------------------------------------------------' in instring or instring[0] == '\n':
        pass
    elif ' ====================================================================' in instring:
        pass
    else:
        natpoplist = instring.split()
        if natpoplist[0][-1].isalpha():
            anum = int(natpoplist[1])-1
            for i in range(0, 4):
                natpop[anum][i] = natpoplist[i+2] #getting the nat pop info from file, index shift of two
        else:
            anumstr  = re.findall(r'\d+', natpoplist[0])[0]
            anum     = int(anumstr)-1
            for i in range(0, 4):
                natpop[anum][i] = natpoplist[i+1] #getting the nat pop info from file, index shift of one (-1 for format)

def orca_parser(file_path, identifier):
    header = ''
    jobstring = ''
    flist = []
    cid = []
    with open(file_path, 'r') as fin:
        counter = 0
        if identifier == 'cid': #this is the identifier type we used
            for line in fin.readlines():
                if 'cid' in line:
                    cid.append(re.findall(r'\d+', line)[1])
                if (counter == 0) and ('$$$$$' in line):
                    header = jobstring
                    jobstring = ''
                    counter += 1

                if 'Timings for individual modules:' in line:
                    flist.append(jobstring)
                    jobstring = ''

                jobstring += line

        elif identifier == 'smiles': #trick to parse commented smiles strings (not used for paper)
            for line in fin.readlines():
                if 'smiles' in line:
                    line = line[line.find('#'):].rstrip()
                    line = line.replace('#smiles:', '')
                    cid.append(line.replace('z', '#'))
                if (counter == 0) and ('$$$$$' in line):
                    header = jobstring
                    jobstring = ''
                    counter += 1

                if 'Timings for individual modules:' in line:
                    flist.append(jobstring)
                    jobstring = ''

                jobstring += line

    orcadic = dict()
    for fnum, i in enumerate(flist):
        fin0 = StringIO()
        fin1 = StringIO()
        fin0.write(header+i)
        fin1.write(header+i)
        fin1.seek(0)

        data      = cclib.io.ccread(fin0)
        rows      = len(data.atomnos)
        columns   = 4
        loeworb   = np.zeros((rows, columns))   #s p d f
        coords    = np.zeros((rows, 3))         # x y z
        natpop    = np.zeros((rows, columns))   # charge core ryd valence
        anboattr = []
        binfo = dict()

        flag     = 0  #flag for orb charges
        nboflag  = 0  #flag for the start of NBO bond info
        anboflag = 0  #flag for the NBO atom info
        natflag  = 0  #flag for natural pop analysis
        counter  = -2 #coord counter (negative for alignment)
        counter1 = -1 #orbital counter (negative for alignment)

        for line in fin1.readlines():
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:   #Z matrice info starts here
                counter = -1
            elif counter == -1:
                if '---------------------------------' in line:
                    counter = 0
            elif counter < rows and counter >= 0:
                match = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*', line)
                
                coords[counter][0] = float(match[0])
                coords[counter][1] = float(match[1])
                coords[counter][2] = float(match[2])
                counter += 1
            elif 'INTERNAL COORDINATES (A.U.)' in line:
                counter = rows+1
            if 'LOEWDIN REDUCED ORBITAL CHARGES' in line:
                counter1 = 0
            if counter1 >= 0 and 's :' in line:
                if flag == 1:
                    counter1 += 1
                match = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*', line.split(':')[-1])
                flag = 1
                loeworb[counter1][0] = float(match[0])
            elif counter1 >= 0 and 'p :' in line:
                match = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*', line.split(':')[-1])
                loeworb[counter1][1] = float(match[0])
            elif counter1 >= 0 and 'd :' in line:
                match = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*', line.split(':')[-1])
                loeworb[counter1][2] = float(match[0])
            elif counter1 >= 0 and 'f :' in line:
                match = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*', line.split(':')[-1])
                loeworb[counter1][3] = float(match[0])
                counter1 += 1
                flag = 0
            if anboflag == 1:
                anbo_reader(line, anboattr)
            if natflag == 1:
                natpop_reader(line, natpop)
            if ' NATURAL BOND ORBITALS (Summary):' in line:
                nboflag = 1
            elif 'NAO Atom No lang   Type(AO)    Occupancy      Energy' in line:
                anboflag = 1
            elif ' Summary of Natural Population Analysis:' in line:
                anboflag = 0
            elif '  Atom No    Charge        Core      Valence    Rydberg      Total' in line:
                natflag  = 1
            elif ' ====================================================================' in line:
                natflag = 0
            elif '$END' in line:
                break
            if nboflag == 1:
                nbo_reader(line, coords, binfo)
        if binfo:
            orcadic[cid[fnum]] = MolData(cid[fnum], data.mult, data.nelectrons, np.array(data.atomcoords), np.array(data.scfenergies), np.array(data.atomnos), np.array(data.atommasses), np.array(data.moenergies), np.array(data.moments), np.array(data.atomcharges['hirshfeld']), binfo, anboattr, natpop)
        else:
            print('failed generating molgraph: '+ cid[fnum])
    return orcadic
