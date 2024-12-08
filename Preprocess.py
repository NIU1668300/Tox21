import networkx as nx
from node2vec import Node2Vec
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
import numpy as np
from rdkit.Chem import rdmolops
import pandas as pd
import os


class Preprocessing:
    
    def __init__(self, data):
        self.data = data
       
    def _compute_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        descriptors = {
            "MolecularWeight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "TPSA": Descriptors.TPSA(mol),
        }
        return descriptors

    def _compute_fingerprints(smiles):
        mol = Chem.MolFromSmiles(smiles)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return {
            "MorganFP": list(morgan_fp),
            "MACCSFP": list(maccs_fp),
        }
    
    def _compute_graph_features(graph):
        features = {
            "AverageDegree": sum(dict(graph.degree()).values()) / len(graph.nodes()),
            "Density": nx.density(graph),
            "ClusteringCoefficient": nx.average_clustering(graph),
        }
        if nx.is_connected(graph):
            features["Diameter"] = nx.diameter(graph)
        else:
            features["Diameter"] = None
        return features

    def _compute_node2vec_embeddings(graph, dimensions=64):
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=50, workers=12, seed=42)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        node_embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])
        graph_embedding = node_embeddings.mean(axis=0)  
        return graph_embedding
    
    def _smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
        graph = nx.from_numpy_array(adjacency_matrix)
        for i, atom in enumerate(mol.GetAtoms()):
            graph.nodes[i]['atom_type'] = atom.GetSymbol()
        return graph
    
    def preprocess(self):
        
        smiles_list = self.data['SMILES'].tolist()
        
        
        if not os.path.exists('data/features.csv'):
            features = []
            for smile in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smile)
                    graph = self._smiles_to_graph(smile)
                    
                    descriptors = self._compute_descriptors(smile)
                    fingerprints = self._compute_fingerprints(smile)
                    graph_features = self._compute_graph_features(graph)
                    node2vec_embeddings = self._compute_node2vec_embeddings(graph)
                    
                    combined_features = {**descriptors, **fingerprints, **graph_features, **{f"Node2Vec{i}": node2vec_embeddings[i] for i in range(len(node2vec_embeddings))}}
                    features.append(combined_features)
                except Exception as e:
                    print(f"Tus muertos pisados {smile}: {e}")

            feature_df = pd.DataFrame(features)
            feature_df.to_csv('data/features.csv', index=False)
        else:
            feature_df = pd.read_csv('data/features.csv')
        
    
    