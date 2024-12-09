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

from DataLoader import DataLoader
from Preprocess import Preprocessing



## Load data
data_loader = DataLoader()
data = data_loader.get_processed_source()
preprocessing = Preprocessing(data)
feature_df = preprocessing.preprocess()
feature_df.to_csv("data/features.csv", index=False)

tox21 = pd.read_csv('data/tox21')
label_columns = ['SR-HSE','NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',\
       'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
targets = tox21[label_columns]


## Impute missing values

morgan = feature_df['MorganFP']
maccs = feature_df['MACCSFP']
feature_df.drop(['MorganFP', 'MACCSFP'], axis=1, inplace=True)

iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
targets = pd.DataFrame(iterative_imputer.fit_transform(targets), columns=targets.columns)
feature_df = pd.DataFrame(iterative_imputer.fit_transform(feature_df), columns=feature_df.columns)

feature_df['MorganFP'] = morgan
feature_df['MACCSFP'] = maccs


## Train model
X,y = feature_df, targets.astype(int)


weights = {}
for column in y.columns:
    class_counts = y[column].value_counts()
    weights[column] = class_counts[0] / class_counts[1] if 1 in class_counts else 1
    

classifiers = []

for column in y.columns:
    
    xgb = XGBClassifier(
        use_label_encoder=False,  
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=weights[column],
        enable_categorical = True
    )
    
    classifiers.append(xgb)

model = MultiOutputClassifier(estimator = XGBClassifier(), n_jobs=-1)
model.estimators_ = classifiers
model.fit(X, y)



## Save model

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)