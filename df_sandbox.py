#!/usr/bin/env python3

import json

import pandas as pd
#from tqdm import tqdm

# Load in term frequencies
with open("./data/term_freqs.json", "r") as handle:
    term_freqs = json.load(handle)

# Load in solution values
with open("./data/baseline_solution.json", "r") as handle:
    solution = json.load(handle)
    
uids = ['0']

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])

#dfs = []
#
#for doc in tqdm(term_freqs):
#    temp_dict = dict({'0': doc[0]}, **doc[1])
#    doc_df = pd.DataFrame([temp_dict], columns=uids)
#    dfs.append(doc_df)
#
#df = pd.concat(dfs)

#df.to_csv("./data/term_freqs.csv")

df = pd.read_csv("./data/term_freqs.csv")
# get a solution for one class
y = []
for doc in term_freqs:
    if 'D000273' in solution[doc[0]]:
        y.append(1)
    else:
        y.append(0)

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np

df = df.fillna(0)
x = df.values

y = np.array(y)

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(x, y)

confirmed_feats = []
counter = 0
for feat in feat_selector.support_:
    if feat == True:
        confirmed_feats.append(uids[counter])
    counter += 1

#for feat in selected_feats:
#    print(uids[feat])
    
feat_selector.ranking_[0:10]

x_filtered = feat_selector.transform(x)


########################################

import xgboost as xgb
import numpy as np

df = pd.read_csv("./data/term_freqs.csv")

train = df[0:10000]
test = df[10000:13000]

y_train = []
for doc in term_freqs[0:10000]:
    if 'D000273' in solution[doc[0]]:
        y_train.append(1)
    else:
        y_train.append(0)

y_test = []
for doc in term_freqs[10000:13000]:
    if 'D000273' in solution[doc[0]]:
        y_test.append(1)
    else:
        y_test.append(0)

train = train.fillna(0)
test = test.fillna(0)
test = test.values
x = train.values

y_train = np.array(y_train)

dtrain = xgb.DMatrix(x, label=y_train)

param = {"max_depth": 6, "eta":.3, "verbosity":2}

bst = xgb.train(param, dtrain)

dtest = xgb.DMatrix(test)

ypred = bst.predict(dtest)

from sklearn import metrics
print(metrics.accuracy_score(y_test, np.round(ypred)))

ypred = list(np.round(ypred))

test = []
for idx in range(len(y_test)):
    if y_test[idx] == 1 and ypred[idx] == 1:
        print("wow great success")
    #test.append([y_test[idx], ypred[idx]])