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

