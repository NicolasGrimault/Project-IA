# Projet AI

A project to predict the faulty water pumps realised by a team of 3 students

https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/


## Dependencies

Python 3.6.5 

```
pip install pandas mlxtend
pip install -U scikit-learn
```

## Running

Just run the file `python.py`, a `resultats.csv` file will be created in the Data folder
```
python .\python.py
```

# Methodology

The only data used are the one from the competition Pump it up   
The AI will train on the training set values.


## Preprocessing data
The description of the data is here : 
https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/

There is some missing values in the data, we started by filling the blanks so they will not interfere with the results.

35% of the construction years are empty (20709 on 59400 rows)    
So we replace each empty row by a random value between -σ and σ (sigma for the standart deviation)

Some values are empty or equal to 0 in the following columns :   
funder, installer, subvillage, scheme_name, scheme_management   
Missings values are always less than 7% except for scheme_management (52 %)  
We choose to consider that the absence of data entry is linked to the state of the pump, as such we decide to replace these empty value by a specific NAN category.

3% of the pump don't have coordinates, to keep it simple we replaced them by the center coordinate of Tanzania. (-6.647724;35.022001)

Then the preprocessing label encoder from sklearn replace the labels by numerical values.


We ignore the following data as we think they are irelevant or repetitive :   
id, num_private, recorded_by, date_recorded, region

## Models

The main goal is to predict the state of a pump : functional, non functional or need repaired.    
It's a classification problem so we started by using a decision Tree and got a score of 0.6987. (Way to go)

Next step we took was to implement a Random forest.
As we are still learning we didn't optimize the model good enough and only improved our score to 0.7112

We did try to implement some other model like Gaussian or some ensemble methods but nothing good came out.

The next step was to read the others approachs in the competition discussion to discover that our forest classifier has not enough estimators (we started with 50 then 100).


So, after some reajusting, here the best model we got 
```python
finalforest_clf = RandomForestClassifier(
    n_estimators=1000, 
    max_features='auto', 
    n_jobs=-1, 
    random_state=42)
```


This competition is a very good introduction to the AI world and we would like to thank you all.