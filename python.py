# %%
# Imports Data
import pandas as pd

train = pd.read_csv('Data/TrainingSetValues.csv', header=0, sep = ",")

# %%
# Display data
train.info()

# %%
status_group = {'functional': 0, 'functional needs repair': 1, 'non functional': 2}
source_class = {'groundwater': 0, 'surface': 1, 'unknown': 2}
extraction_type = { 'gravity': 0, 'submersible':1,'swn 80':2,'nira/tanira':3,'india mark ii':4,'other':5,'ksb':6,'mono':7,'windmill':8,'afridev':9,'other - rope pump':10,'india mark iii':11,'other - swn 81':12,'other - play pump':13,'cemo':14,'climax':15,'walimi':16,'other - mkulima/shinyanga':17}

# %%
# Format Data
train['status_group'] = train['status_group'].map(status_group).astype(int)
train['source_class'] = train['source_class'].map(source_class).astype(int)
train['extraction_type'] = train['extraction_type'].map(extraction_type).astype(int)

# %%
# Drop useless data
train = train.drop(["id","num_private","recorded_by"], axis = 1)

# %%
# Replace date
import numpy as np

foo = np.array(train['construction_year'])
avg = np.mean(foo[foo > 0])
std = np.std(foo[foo > 0])
count = len(foo[foo == 0])
random_list = np.random.randint(avg - std, avg + std, size=count)
train.loc[train['construction_year'] == 0,'construction_year'] = random_list

#%%
# Split data to train and test
from sklearn.model_selection import train_test_split

X_alltrain = train.values[:,:36]
y_alltrain = train.values[:, 37]
X_train, X_test, y_train, y_test = train_test_split(X_alltrain, y_alltrain, random_state=42)

# %%
# Define model
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train,y_train)