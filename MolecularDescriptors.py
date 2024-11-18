import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors


class MolecularDescriptors:
    def __init__(self, smiles):
        self.smiles = smiles
        self.descritors = Descriptors._desclist
        
    def get_descriptors(self):
        
        res = np.zeros((len(self.smiles), len(self.descriptors)))
        for i, smile in enumerate(tqdm(self.smiles)):
            mol = Chem.MolFromSmiles(smile)
            for j, (name, fun) in enumerate(self.descriptors):
                try:
                    value = fun(mol)
                except:
                    value = np.nan
                res[i][j] = value
        return res