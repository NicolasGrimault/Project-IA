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

# %%
# Process specific data

foo = np.array(train['construction_year'])
avg = np.mean(foo[foo > 0])
std = np.std(foo[foo > 0])
count = len(foo[foo == 0])
random_list = np.random.randint(avg - std, avg + std, size=count)
train.loc[train['construction_year'] == 0,'construction_year'] = random_list


specificHeaders = ['funder','installer','subvillage','scheme_name','scheme_management']
for header in specificHeaders:
    train[header] = train[header].fillna('NAN')


train.loc[train['funder'] == '0', 'funder'] = 'NAN'
train.loc[train['installer'] == '0', 'installer'] = 'NAN'
train.loc[train['longitude'] == 0, 'latitude'] = -6.647724
train.loc[train['longitude'] == 0, 'longitude'] = 35.022001

#train['longitude'] = train['longitude'].astype(float)

# %%
#Process string data
headers = ['funder','installer','wpt_name','basin','subvillage','lga','ward','public_meeting','scheme_name','permit','scheme_management','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class','waterpoint_type','waterpoint_type_group','status_group']

for header in headers:
    train[header] = lab_enc.fit_transform(train[header])

train = train.drop(["id","num_private","recorded_by", "date_recorded", "region"], axis = 1)


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
