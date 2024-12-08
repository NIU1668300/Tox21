import pandas as pd
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
preprocessing.preprocess()

tox21 = pd.read_csv('data/tox21')
label_columns = ['SR-HSE','NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',\
       'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
feature_df = pd.read_csv('data/features.csv')
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


## Classify
X,y = feature_df, targets.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


weights = {}
for column in y_train.columns:
    class_counts = y_train[column].value_counts()
    weights[column] = class_counts[0] / class_counts[1] if 1 in class_counts else 1
    

classifiers = []

for column in y_train.columns:
    
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
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## Show results

f1_scores = []
for i, column in enumerate(y.columns):
    f1 = f1_score(y_test.iloc[:, i], y_pred[:, i], average="macro", zero_division=0)
    f1_scores.append(f1)
    print(f"F1-Score for {column}: {f1:.4f}")
    
mean_f1 = sum(f1_scores) / len(f1_scores)
print(f"\nMean F1-Score across all outputs: {mean_f1:.4f}")

mat = multilabel_confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.ravel()

for i, (ax, label) in enumerate(zip(axes, label_columns)):
    sns.heatmap(mat[i], annot=True, fmt='d', ax=ax, cmap='coolwarm', cbar=False)
    ax.set_title(label)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()


