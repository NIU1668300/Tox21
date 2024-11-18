import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools
import re
from tqdm import tqdm
from functools import reduce

class DataLoader:
    
    def __init__(self ):
        self.source = PandasTools.LoadSDF('data/tox21_10k_data_all.sdf', smilesName='SMILES', molColName='Molecule', includeFingerprints=True)
        
    def clean_number(self, s):
        return float(re.split(r'[\s(]', s)[0])
    
    def add(self, x, y):
        if np.isnan(x) and np.isnan(y):
            return x
        elif not np.isnan(x) and np.isnan(y):
            return x
        elif np.isnan(x) and not np.isnan(y):
            return y
        elif x == y:
            return x
        else:
            return 1.0
    
    def add_pd(self, series):
        return reduce(lambda x,y: self.add(x, y), series)
    
    def find_duplicates(self, column_name):
        return self.source[column_name][self.source[column_name].duplicated()].values
    
    def change_types(self, column_names, type):
        self.source = self.source.astype({c: type for c in column_names})
        
    def clean_numbers(self, column_names):
        for c in column_names:
            self.source[c] = self.source[c].apply(lambda x: self.clean_number(x))
    
    def merge_duplicate_rows(self, duplicate_column, target_columns):
        duplicate_rows = self.find_duplicates(duplicate_column)
        
        for d in tqdm(duplicate_rows):
            
            temp = self.source[self.source[duplicate_column] == d][target_columns].apply(self.add_pd)
            indx = list(self.source[self.source[duplicate_column] == d].index)
            keep_i = indx[0]
            drop_i = indx[1:]
            
            temp2 = self.source.loc[keep_i]
            temp2.update(temp)
            self.source.loc[keep_i] = temp2
            
            for i in drop_i:
                self.source = self.source.drop(i)
    
    def save(self):
        self.source.to_csv('data/tox21_10k_data_all.csv')
        
    def get_source(self):
        return self.source 
    
    def override_source(self, new_source):
        self.source = new_source
    
    def get_processed_source(self):
        #Revisar bien esta funcion
        target_columns = ['SR-HSE', 'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 
                  'SR-MMP', 'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
        self.clean_numbers(['FW'])
        self.change_types(target_columns, 'float')
        self.merge_duplicate_rows('DSSTox_CID', target_columns)
        return self.source
