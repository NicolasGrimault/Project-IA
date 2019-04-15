#%%
#Imports Data
import pandas as pd

train = pd.read_csv('Data/TrainingSetValues.csv', header = 0)

#%% 
#Display data 
train.info()
