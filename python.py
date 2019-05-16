# %%
# Imports Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

lab_enc = preprocessing.LabelEncoder()

train = pd.read_csv('Data/TrainingSetValues.csv', header=0, sep = ",")
#train.info()

# %%
# Display data
train.info()

# %%
train['funder'] = train['funder'].fillna('NAN')
train.loc[train['funder'] == '0', 'funder'] = 'NAN'
train['installer'] = train['installer'].fillna('NAN')
train.loc[train['installer'] == '0', 'installer'] = 'NAN'
train['subvillage'] = train['subvillage'].fillna('NAN')
train['scheme_name'] = train['scheme_name'].fillna('NAN')
train['scheme_management'] = train['scheme_management'].fillna('NAN')

train.loc[train['longitude'] == 0, 'latitude'] = -6.647724
train.loc[train['longitude'] == 0, 'longitude'] = 35.022001

train['funder'] = lab_enc.fit_transform(train['funder'])
train['installer'] = lab_enc.fit_transform(train['installer'])
train['wpt_name'] = lab_enc.fit_transform(train['wpt_name'])
train['basin'] = lab_enc.fit_transform(train['basin'])
train['subvillage'] = lab_enc.fit_transform(train['subvillage'])
train['lga'] = lab_enc.fit_transform(train['lga'])
train['ward'] = lab_enc.fit_transform(train['ward'])
train['public_meeting'] = lab_enc.fit_transform(train['public_meeting'])
train['scheme_name'] = lab_enc.fit_transform(train['scheme_name'])
train['permit'] = lab_enc.fit_transform(train['permit'])
train['scheme_management'] = lab_enc.fit_transform(train['scheme_management'])
train['extraction_type'] = lab_enc.fit_transform(train['extraction_type'])
train['extraction_type_group'] = lab_enc.fit_transform(train['extraction_type_group'])
train['extraction_type_class'] = lab_enc.fit_transform(train['extraction_type_class'])
train['management'] = lab_enc.fit_transform(train['management'])
train['management_group'] = lab_enc.fit_transform(train['management_group'])
train['payment'] = lab_enc.fit_transform(train['payment'])
train['payment_type'] = lab_enc.fit_transform(train['payment_type'])
train['water_quality'] = lab_enc.fit_transform(train['water_quality'])
train['quality_group'] = lab_enc.fit_transform(train['quality_group'])
train['quantity'] = lab_enc.fit_transform(train['quantity'])
train['quantity_group'] = lab_enc.fit_transform(train['quantity_group'])
train['source'] = lab_enc.fit_transform(train['source'])
train['source_type'] = lab_enc.fit_transform(train['source_type'])
train['source_class'] = lab_enc.fit_transform(train['source_class'])
train['waterpoint_type'] = lab_enc.fit_transform(train['waterpoint_type'])
train['waterpoint_type_group'] = lab_enc.fit_transform(train['waterpoint_type_group'])
train['status_group'] = lab_enc.fit_transform(train['status_group'])
#.astype(str)
train['longitude'] = train['longitude'].astype(float)


train = train.drop(["id","num_private","recorded_by", "date_recorded", "region"], axis = 1)


#train.info()

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
