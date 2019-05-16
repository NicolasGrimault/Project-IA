# %%
# Imports Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('Data/TrainingSetValues.csv', header=0, sep = ",")
#train.info()

# %%
# Display data
train.info()

# %%
train['funder'] = lab_enc.fit_transform(train['funder'].astype(str))
train['installer'] = lab_enc.fit_transform(train['installer'].astype(str))
train['wpt_name'] = lab_enc.fit_transform(train['wpt_name'].astype(str))
train['basin'] = lab_enc.fit_transform(train['basin'].astype(str))
train['subvillage'] = lab_enc.fit_transform(train['subvillage'].astype(str))
train['lga'] = lab_enc.fit_transform(train['lga'].astype(str))
train['ward'] = lab_enc.fit_transform(train['ward'].astype(str))
train['public_meeting'] = lab_enc.fit_transform(train['public_meeting'].astype(str))
train['scheme_name'] = lab_enc.fit_transform(train['scheme_name'].astype(str))
train['permit'] = lab_enc.fit_transform(train['permit'].astype(str))
train['scheme_management'] = lab_enc.fit_transform(train['scheme_management'].astype(str))
train['extraction_type'] = lab_enc.fit_transform(train['extraction_type'].astype(str))
train['extraction_type_group'] = lab_enc.fit_transform(train['extraction_type_group'].astype(str))
train['extraction_type_class'] = lab_enc.fit_transform(train['extraction_type_class'].astype(str))
train['management'] = lab_enc.fit_transform(train['management'].astype(str))
train['management_group'] = lab_enc.fit_transform(train['management_group'].astype(str))
train['payment'] = lab_enc.fit_transform(train['payment'].astype(str))
train['payment_type'] = lab_enc.fit_transform(train['payment_type'].astype(str))
train['water_quality'] = lab_enc.fit_transform(train['water_quality'].astype(str))
train['quality_group'] = lab_enc.fit_transform(train['quality_group'].astype(str))
train['quantity'] = lab_enc.fit_transform(train['quantity'].astype(str))
train['quantity_group'] = lab_enc.fit_transform(train['quantity_group'].astype(str))
train['source'] = lab_enc.fit_transform(train['source'].astype(str))
train['source_type'] = lab_enc.fit_transform(train['source_type'].astype(str))
train['source_class'] = lab_enc.fit_transform(train['source_class'].astype(str))
train['waterpoint_type'] = lab_enc.fit_transform(train['waterpoint_type'].astype(str))
train['waterpoint_type_group'] = lab_enc.fit_transform(train['waterpoint_type_group'].astype(str))
train['status_group'] = lab_enc.fit_transform(train['status_group'].astype(str))

train['longitude'] = train['longitude'].astype(float)

train = train.drop(["id","num_private","recorded_by", "date_recorded", "region"], axis = 1)


train.info()

# %%
# Replace date
foo = np.array(train['construction_year'])
avg = np.mean(foo[foo > 0])
std = np.std(foo[foo > 0])
count = len(foo[foo == 0])
random_list = np.random.randint(avg - std, avg + std, size=count)
train.loc[train['construction_year'] == 0,'construction_year'] = random_list

#%%
# Split data to train and test
X_alltrain = train.values[:,:train.shape[1]-2]
y_alltrain = train.values[:, train.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X_alltrain, y_alltrain, random_state=42)


# %%
# Define model
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train,y_train)

y_pred = tree_clf.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(score*100)
