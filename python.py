# %%
# Imports Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

lab_enc = preprocessing.LabelEncoder()

train = pd.read_csv('Data/TrainingSetValues.csv', header=0, sep=",")
Test = pd.read_csv('Data/TestSetValues.csv', header=0, sep=",")
Testindex = Test.id
full_data = [train, Test]


def generer_resultats(clf, data=Test.values, idx=Testindex):
    clf.fit(X_train, y_train)
    prediction = clf.predict(data)
    results = pd.DataFrame(prediction.astype(
        int), index=idx, columns=['status_group'])
    results['status_group'] = lab_enc.inverse_transform(
        results['status_group'])
    results.to_csv('./Data/resultats.csv')

# %%
# Process specific data


for data in full_data:
    npa = np.array(data['construction_year'])
    avg = np.mean(npa[npa > 0])
    std = np.std(npa[npa > 0])
    count = len(npa[npa == 0])
    random_list = np.random.randint(avg - std, avg + std, size=count)
    data.loc[data['construction_year'] ==
             0, 'construction_year'] = random_list

    specificHeaders = ['funder', 'installer',
                       'subvillage', 'scheme_name', 'scheme_management']
    for header in specificHeaders:
        data[header] = data[header].fillna('NAN')

    data.loc[data['funder'] == '0', 'funder'] = 'NAN'
    data.loc[data['installer'] == '0', 'installer'] = 'NAN'
    data.loc[data['longitude'] == 0, 'latitude'] = -6.647724
    data.loc[data['longitude'] == 0, 'longitude'] = 35.022001

# %%
# Process string data


headers = ['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'lga', 'ward', 'public_meeting', 'scheme_name', 'permit', 'scheme_management', 'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group']
for header in headers:
    Test[header] = lab_enc.fit_transform(Test[header])

Test = Test.drop(["id", "num_private", "recorded_by",
                        "date_recorded", "region"], axis=1)

# As the label encoder is used later to deencode status_group, the last one used must be status_group
headers = ['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'lga', 'ward', 'public_meeting', 'scheme_name', 'permit', 'scheme_management', 'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group', "status_group"]
for header in headers:
    train[header] = lab_enc.fit_transform(train[header])

train = train.drop(["id", "num_private", "recorded_by",
                    "date_recorded", "region"], axis=1)


# %%
# Split data to train and test
X_alltrain = train.values[:, :train.shape[1]-1]
y_alltrain = train.values[:, train.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(
    X_alltrain, y_alltrain, random_state=42)

# %%
# Forest
finalforest_clf = RandomForestClassifier(
    n_estimators=1000, 
    max_features='auto', 
    n_jobs=-1, 
    random_state=42)
finalforest_clf.fit(X_train, y_train)
y_pred = finalforest_clf.predict(X_test)
print("Forest" + str(accuracy_score(y_test, y_pred)*100))
    

generer_resultats(finalforest_clf, Test)
print('END')
