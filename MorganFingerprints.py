import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class MorganFingerprints:
    def __init__(self, smiles):
        self.smiles = smiles
        
    def get_fingerprint_from_smile(self, smile, r):
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, r)
        self.fingerprints.append(fp)
        return np.array(list(fp))
    
    def transform(self, r):
        res = []
        self.fingerprints = []
        
        for smile in self.smiles:
            res.append(self.get_fingerprint_from_smile(smile, r))
            
        return np.array(res)
