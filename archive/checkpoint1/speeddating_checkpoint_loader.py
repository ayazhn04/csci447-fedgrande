import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATASET_ID=40536
SEED=42
N_SPLITS=5

# load the same OpenML dataset used in the GRANDE benchmark
ds=openml.datasets.get_dataset(DATASET_ID,download_data=True)
X,y,categorical_indicator,feature_names=ds.get_data(
    target=ds.default_target_attribute,
    dataset_format='dataframe'
)

# combine predictors and target only for display/checking
df=X.copy()
df[ds.default_target_attribute]=y

print('=== DATASET ===')
print('OpenML ID:',DATASET_ID)
print('Name:',ds.name)
print('Version:',ds.version)
print('Target:',ds.default_target_attribute)
print('X shape:',X.shape)
print('y shape:',y.shape)
print('Combined X+y shape:',df.shape)

print('\n=== FIRST FIVE ROWS (X + target) ===')
print(df.head(5).to_csv(index=False),end='')

print('\n=== CLASS DISTRIBUTION ===')
y_str=y.astype(str)
counts=y_str.value_counts().sort_index()
print(counts.to_string())
print('Total:',len(y_str))
for cls,n in counts.items():
    print(f'class {cls}: {n} ({100*n/len(y_str):.12f}%)')
minority_pct=100*counts.min()/len(y_str)
print(f'Minority percentage: {minority_pct:.14f}%')
print(f'Difference from paper 16.47%: {minority_pct-16.47:.14f} percentage points')

print('\n=== MISSING VALUES ===')
missing=X.isna().sum()
missing=missing[missing>0]
print('Total missing predictor cells:',int(missing.sum()))
print('Predictor columns with missing values:',len(missing))
print(missing.to_string())
print('Target missing values:',int(y.isna().sum()))

print('\n=== OUTER SPLIT CHECK ===')
# The paper generates shuffled stratified folds at runtime; there are no static split files.
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
test_seen=np.zeros(len(X),dtype=int)
for fold,(train_idx,test_idx) in enumerate(skf.split(X,y),start=1):
    overlap=np.intersect1d(train_idx,test_idx).size
    test_seen[test_idx]+=1
    print(f'fold {fold}: train={len(train_idx)}, test={len(test_idx)}, train/test overlap={overlap}')
print('Samples held out exactly once across the 5 outer folds:',int((test_seen==1).sum()))
print('Samples held out more than once:',int((test_seen>1).sum()))
print('Samples never held out:',int((test_seen==0).sum()))
