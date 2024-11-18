from rdkit.Chem.Scaffolds import MurckoScaffold as MS

class MurckoScaffold:
    def __init__(self, smiles):
        self.smiles = smiles
        
    def get_scaffold(self, smile):
        scaffold = MS.MurckoScaffoldSmilesFromSmiles(smile)
        return scaffold if scaffold else smile
    
    def transform(self):
        sc_smiles = []
        for smile in self.smiles:
            sc_smiles.append(self.get_scaffold(smile))
        return sc_smiles
    
    