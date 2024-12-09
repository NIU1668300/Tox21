import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, f1_score
from colorama import Fore, Style

from DataLoader import DataLoader
from Preprocess import Preprocessing

try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model not found please run train.py first or save your own model as 'models/model.pkl'")
    
model.enable_categorical = True
        

parser = argparse.ArgumentParser(description='Predict the toxicity of a molecule')
parser.add_argument('smiles', type=str, help='The SMILES representation of the molecule')

args = parser.parse_args()

X = pd.DataFrame([args.smiles], columns=['SMILES'])
preprocessing = Preprocessing(X)
feature_df = preprocessing.preprocess()



feature_df.drop(['MorganFP', 'MACCSFP'], axis=1, inplace=True)

y_pred = model.predict(feature_df)


print("*"*50)
print(f"Predicted toxicity of the molecule {args.smiles}:")
label_columns = ['SR-HSE','NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',\
       'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
for name, result in zip(label_columns, y_pred[0]):
    print(f"{name}: ", end = '')
    if result:
        print(Fore.GREEN + "Active")
    else:
        print(Fore.RED + "Inactive")
    print(Style.RESET_ALL, end = '')