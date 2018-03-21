
'''

QUOVO Code Challenge
Author: Shadie Khubba
March 11, 2018

Summary: Achieved high accuracy levels simply through pre-processing,
         feature creation, and a Random Forest model with minimal tuning.

'''

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import re
import string

#######################################################################
# Pre-processing
#######################################################################

# Read in the data
train = pd.DataFrame.from_csv('code_challenge_train.csv')
test = pd.DataFrame.from_csv('code_challenge_test.csv')

# Reset index 
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Balance check: ~ 0.40 of train cases match, similar proportion in test df
sum(train['x_id'] == train['y_id'])/len(train)
sum(test['x_id'] == test['y_id'])/len(test)

# Check for null values: none found
train.isnull().sum()
test.isnull().sum()

# Split into feature set and response
x_train = train[['x_description', 'y_description']]
x_test = test[['x_description', 'y_description']]

y_train = train['x_id'] == train['y_id']
y_test = test['x_id'] == test['y_id']

# Convert training sets to lowercase and remove punctuation
def tidy(text):
    text = text.lower()
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return(text)

x_train = x_train.applymap(tidy)
x_test = x_test.applymap(tidy)

# Remove commonly used terms (with regex), extra spaces
word_list = ['(^| )inc($| )','(^| )incorporated($| )', '(^| )fund($| )', '(^| )fd($| )', '(^| )ltd($| )', \
             '(^| )limited($| )', '(^| )corporation($| )', '(^| )corp($| )', '(^| )holdings($| )', \
             '(^| )trust($| )', '(^| )company($| )', '(^| )co($| )', '(^| )international($| )',  \
             '(^| )the($| )',  '(^| )class($| )',  '(^| )cl($| )',  '(^| )international($| )']

def remove_words(text, word_list):
    words = '|'.join(word_list)
    text = re.sub(words, ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub('(^ | $)', '', text)
    return(text)
    
x_train = x_train.applymap(lambda x: remove_words(x, word_list))
x_test = x_test.applymap(lambda x: remove_words(x, word_list))

#######################################################################
# Feature creation
#######################################################################

# Absolute value of the difference in the number of space-delimited terms of x,y
x_train['num_words'] = x_train.apply(lambda row: abs(len(str.split(row['x_description'])) - \
       len(str.split(row['y_description']))), axis = 1)
x_test['num_words'] = x_test.apply(lambda row: abs(len(str.split(row['x_description'])) - \
      len(str.split(row['y_description']))), axis = 1)

# Dummy variable for: does the first term match perfectly?
x_train['first_match'] = np.array(x_train.apply(lambda row: re.search('^[^\s]+', row['x_description']).group() == \
       re.search('^[^\s]+', row['y_description']).group(), axis = 1), dtype = 'int')
x_test['first_match'] = np.array(x_test.apply(lambda row: re.search('^[^\s]+', row['x_description']).group() == \
      re.search('^[^\s]+', row['y_description']).group(), axis = 1), dtype = 'int')

########################################################
# Feature creation using fuzzywuzzy package
# https://pypi.python.org/pypi/fuzzywuzzy
########################################################

# Partial ratio
x_train['partial'] = x_train.apply(lambda row: fuzz.partial_ratio(row['x_description'], \
       row['y_description']), axis = 1)
x_test['partial'] = x_test.apply(lambda row: fuzz.partial_ratio(row['x_description'], \
      row['y_description']), axis = 1)

# Partial token set ratio
x_train['ptsr'] = x_train.apply(lambda row: fuzz.partial_token_set_ratio(row['x_description'], \
       row['y_description']), axis = 1)
x_test['ptsr'] = x_test.apply(lambda row: fuzz.partial_token_set_ratio(row['x_description'], \
      row['y_description']), axis = 1)

# Token sort
x_train['token_sort'] = x_train.apply(lambda row: fuzz.token_sort_ratio(row['x_description'], \
       row['y_description']), axis = 1)
x_test['token_sort'] = x_test.apply(lambda row: fuzz.token_sort_ratio(row['x_description'], \
      row['y_description']), axis = 1)

# Token set
x_train['token_set'] = x_train.apply(lambda row: fuzz.token_set_ratio(row['x_description'], \
       row['y_description']), axis = 1)
x_test['token_set'] = x_test.apply(lambda row: fuzz.token_set_ratio(row['x_description'], \
      row['y_description']), axis = 1)

# Ratio
x_train['ratio'] = x_train.apply(lambda row: fuzz.ratio(row['x_description'], row['y_description']), axis = 1)
x_test['ratio'] = x_test.apply(lambda row: fuzz.ratio(row['x_description'], row['y_description']), axis = 1)

# Partial ratio for series of first letters in each term
def firstletter(x):
    L = str.split(x)
    newstring = ' '.join([item[:1] for item in L])
    return(newstring)
    
x_train['first_letter'] = x_train.apply(lambda row: fuzz.partial_ratio(firstletter(row['x_description']), \
       firstletter(row['y_description'])), axis = 1)
x_test['first_letter'] = x_test.apply(lambda row: fuzz.partial_ratio(firstletter(row['x_description']), \
      firstletter(row['y_description'])), axis = 1)

# Token set ratio for string with no vowels
def novowels(x):
    return(''.join([l for l in x if l not in ['a','e','i','o','u']]))

x_train['no_vowels'] = x_train.apply(lambda row: fuzz.token_set_ratio(novowels(row['x_description']), \
       novowels(row['y_description'])), axis = 1)
x_test['no_vowels'] = x_test.apply(lambda row: fuzz.token_set_ratio(novowels(row['x_description']), \
      novowels(row['y_description'])), axis = 1)

# Token set ratio for first term
x_train['first_ratio'] = x_train.apply(lambda row: fuzz.token_set_ratio(re.search('^[^\s]+', row['x_description']).group(), \
       re.search('^[^\s]+', row['y_description']).group()), axis = 1)
x_test['first_ratio'] = x_test.apply(lambda row: fuzz.token_set_ratio(re.search('^[^\s]+', row['x_description']).group(), \
       re.search('^[^\s]+', row['y_description']).group()), axis = 1)

# Partial ratio for series of first + last letters in each term
def fl_letter(x):
    L = str.split(x)
    newstring = ' '.join([item[:1]+item[-1:] if len(item) != 1 else item[-1:] for item in L])
    return(newstring)
    
x_train['first_last'] = x_train.apply(lambda row: fuzz.partial_ratio(fl_letter(row['x_description']), \
       fl_letter(row['y_description'])), axis = 1)
x_test['first_last'] = x_test.apply(lambda row: fuzz.partial_ratio(fl_letter(row['x_description']), \
      fl_letter(row['y_description'])), axis = 1)

#######################################################################
# Modeling
#######################################################################

# Omitting the full strings
x_train = x_train.iloc[:,2:]
x_test = x_test.iloc[:,2:]

# Fit the model

model = RandomForestClassifier(n_estimators = 400, random_state = 0)
model.fit(x_train, y_train)

# Train score
model.score(x_train, y_train)

#######################################################################
# Results
#######################################################################

# Train confusion matrix
confusion_matrix(y_train,model.predict(x_train))/len(y_train)

# Test confusion matrix: 43 / 4217 misclassified
confusion_matrix(y_test,model.predict(x_test))/len(y_test)

# Output exceptions for further review
miss = np.ravel(np.where(np.array(y_test != model.predict(x_test))))
miss_feat = test.loc[miss,:]
miss_y = y_test.loc[miss]

output = pd.concat([miss_feat,miss_y], axis = 1)
output.to_csv("output.csv")
