# %%
# Imports Data
import pandas as pd

train = pd.read_csv('Data/TrainingSetValues.csv', header=0, sep = ",")

# %%
# Display data
train.info()

# %%
train['status_group'] = train['status_group'].map({'functional': 0, 'functional needs repair': 0.5, 'non functional': 1}).astype(float)
train['source_class'] = train['source_class'].map({'groundwater': 0, 'surface': 1, 'unknown': 2}).astype(int)
train['extraction_type'] = train['extraction_type'].map({ 'gravity': 0, 'submersible':1,'swn 80':2,'nira/tanira':3,'india mark ii':4,'other':5,'ksb':6,'mono':7,'windmill':8,'afridev':9,'other - rope pump':10,'india mark iii':11,'other - swn 81':12,'other - play pump':13,'cemo':14,'climax':15,'walimi':16,'other - mkulima/shinyanga':17}).astype(int)
train['water_quality'] = train['water_quality'].map({'coloured': 0, 'fluoride': 1, 'fluoride abandoned': 2, 'milky': 3, 'salty': 4, 'salty abandoned': 5, 'soft': 6, 'unknown': 7}).astype(int)
train['quantity'] = train['quantity'].map({'dry': 0, 'enough': 1, 'insufficient': 2, 'seasonal': 3, 'unknown': 4}).astype(int)
train['quantity_group'] = train['quantity_group'].map({'dry': 0, 'enough': 1, 'insufficient': 2, 'seasonal': 3, 'unknown': 4}).astype(int)
train['source'] = train['source'].map({'dam': 0, 'hand dtw': 1, 'lake': 2, 'machine dbh': 3, 'other': 4, 'rainwater harvesting': 5, 'river': 6, 'shallow well': 7, 'spring': 8, 'unknown': 9}).astype(int)

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
