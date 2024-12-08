# Tox21

This repository is a Machine Learning project that tackles the Tox-21 data challenge in its reduced version, the data can be obtained from its webpage:

https://tripod.nih.gov/tox21/challenge/data.jsp

In this repository a Machine Learning model is created to predict the response panel of 12 toxicity endpoints in drug-related molecules done making feature creation and engineering using computational chemistry libraries.

# Data

The used data consists on SMILES representations of 8039 molecules along with the experimental response panel. For illustration the last five rows of the dataframe are:

 | SMILES                                |   SR-HSE |   NR-AR |   SR-ARE |   NR-Aromatase |   NR-ER-LBD |   NR-AhR |   SR-MMP |   NR-ER |   NR-PPAR-gamma |   SR-p53 |   SR-ATAD5 |   NR-AR-LBD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CCCc1cc(=O)[nH]c(=S)[nH]1             |        0 |       0 |        0 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 
| S=C1NCCN1                             |        0 |       0 |        1 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 |
| S=C1NCCN1                             |        0 |       0 |        0 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 |
| CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1  |        0 |       0 |        0 |              0 |           0 |        1 |        0 |       0 |               0 |        0 |          0 |           0 |
| CCC(COC(=O)CCS)(COC(=O)CCS)COC(=O)CCS |        0 |       0 |        0 |              0 |           0 |        0 |   0 |       0 |               0 |        1 |          0 |           0 |


A SMILES (Simplified Molecular Input Line Entry System) representation refers to a "short" ASCII string that represents the structure of a chemical species. For instance the SMILES CCC(COC(=O)CCS)(COC(=O)CCS)COC(=O)CCS represents

![image](/assets/Molecule.png)

# Project outline

**Exploratory data analysis** which can be found on `EDA.ipynb` serves the purpose of finding an initial description of the data we are working with 