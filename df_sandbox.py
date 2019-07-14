"""
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

    

##################### rf - f1 = .57 w/ n_estimators=100 for D000818

term_to_test = "D056128"

import json
import numpy as np
import time

# Load in term frequencies
with open("./data/term_freqs_rev_0.json", "r") as handle:
    temp = json.load(handle)
 
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])
        
docs_list = list(temp.keys())
partition = int(len(docs_list) * .8)

train_docs = docs_list[0:partition]
test_docs = docs_list[partition:]

# Load in solution values
solution = {}
docs_list = set(docs_list)
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:
            solution[line[0]] = line[1:]

train_docs = [doc for doc in train_docs if doc in solution.keys()]
test_docs = [doc for doc in test_docs if doc in solution.keys()]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]
start = time.perf_counter()
y = []
for doc in train_docs:
    if term_to_test in solution[doc]:
        y.append(1)
    else:
        y.append(0)

#y = np.array(y)

x_0 = []
x_1 = []
for doc in train_docs:
    row = []
    for uid in uids:
        if uid in train_freqs[doc].keys():
            row.append(train_freqs[doc][uid])
        else:
            row.append(0)
    row.append(y[train_docs.index(doc)])
    if row[-1] == 0:
        x_0.append(row)
    else:
        x_1.append(row)

x_0 = np.array(x_0)
x_0 = x_0[0:20000]

x_1 = np.array(x_1)

from sklearn.utils import resample

x_1 = resample(x_1, replace=True, n_samples=int(len(x_0) * .66), random_state=42)

x = np.vstack((x_0, x_1))

del(x_1)
del(x_0)

y = x[:,-1]
x = x[:,:-1]

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(verbose=1, random_state=42, n_estimators=100)

clf.fit(x, y)
print("elapsed: " + str(time.perf_counter() - start))
test_x = []
for doc in test_freqs.keys():
    row = []
    for uid in uids:
        if uid in test_freqs[doc].keys():
            row.append(test_freqs[doc][uid])
        else:
            row.append(0)
    test_x.append(row)

test_x = np.array(test_x)

test_y = []
for doc in test_freqs.keys():
    if term_to_test in solution[doc]:
        test_y.append(1)
    else:
        test_y.append(0)
        
test_y = np.array(test_y)

predictions = clf.predict(test_x)

clf.score(test_x, test_y)

true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0

for idx in range(len(test_y)):
    if predictions[idx] == 1 and test_y[idx] == 1:
        true_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 1:
        false_neg += 1
    if predictions[idx] == 1 and test_y[idx] == 0:
        false_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 0:
        true_neg += 1

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
f1 = (2 * precision * recall) / (precision + recall)

##################### f1 - .479

import json
import numpy as np

# Load in term frequencies
with open("./data/term_freqs_rev_0.json", "r") as handle:
    temp = json.load(handle)
 
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])
        
docs_list = list(temp.keys())
partition = int(len(docs_list) * .8)

train_docs = docs_list[0:partition]
test_docs = docs_list[partition:]

# Load in solution values
solution = {}
docs_list = set(docs_list)
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:
            solution[line[0]] = line[1:]

train_docs = [doc for doc in train_docs if doc in solution.keys()]
test_docs = [doc for doc in test_docs if doc in solution.keys()]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]

y = []
for doc in train_docs:
    if "D000818" in solution[doc]:
        y.append(1)
    else:
        y.append(0)

#y = np.array(y)

x_0 = []
x_1 = []
for doc in train_docs:
    row = []
    for uid in uids:
        if uid in train_freqs[doc].keys():
            row.append(train_freqs[doc][uid])
        else:
            row.append(0)
    row.append(y[train_docs.index(doc)])
    if row[-1] == 0:
        x_0.append(row)
    else:
        x_1.append(row)

x_0 = np.array(x_0)
x_0 = x_0[0:20000]

x_1 = np.array(x_1)

from sklearn.utils import resample

x_1 = resample(x_1, replace=True, n_samples=int(len(x_0) * .66), random_state=42)

x = np.vstack((x_0, x_1))

del(x_1)
del(x_0)

y = x[:,-1]
x = x[:,:-1]

from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(x, y)

test_x = []
for doc in test_freqs.keys():
    row = []
    for uid in uids:
        if uid in test_freqs[doc].keys():
            row.append(test_freqs[doc][uid])
        else:
            row.append(0)
    test_x.append(row)

test_x = np.array(test_x)

test_y = []
for doc in test_freqs.keys():
    if "D000818" in solution[doc]:
        test_y.append(1)
    else:
        test_y.append(0)
        
test_y = np.array(test_y)

predictions = clf.predict(test_x)

clf.score(test_x, test_y)

true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0

for idx in range(len(test_y)):
    if predictions[idx] == 1 and test_y[idx] == 1:
        true_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 1:
        false_neg += 1
    if predictions[idx] == 1 and test_y[idx] == 0:
        false_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 0:
        true_neg += 1

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
f1 = (2 * precision * recall) / (precision + recall)

####################### xgb

import json
import numpy as np

# Load in term frequencies
with open("./data/term_freqs_rev_0.json", "r") as handle:
    temp = json.load(handle)
 
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        uids.append(line[0])
        
docs_list = list(temp.keys())
partition = int(len(docs_list) * .8)

train_docs = docs_list[0:partition]
test_docs = docs_list[partition:]

# Load in solution values
solution = {}
docs_list = set(docs_list)
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:
            solution[line[0]] = line[1:]

train_docs = [doc for doc in train_docs if doc in solution.keys()]
test_docs = [doc for doc in test_docs if doc in solution.keys()]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]

y = []
for doc in train_docs:
    if "D000818" in solution[doc]:
        y.append(1)
    else:
        y.append(0)

#y = np.array(y)

x_0 = []
x_1 = []
for doc in train_docs:
    row = []
    for uid in uids:
        if uid in train_freqs[doc].keys():
            row.append(train_freqs[doc][uid])
        else:
            row.append(0)
    row.append(y[train_docs.index(doc)])
    if row[-1] == 0:
        x_0.append(row)
    else:
        x_1.append(row)

x_0 = np.array(x_0)
x_0 = x_0[0:20000]

x_1 = np.array(x_1)

from sklearn.utils import resample

x_1 = resample(x_1, replace=True, n_samples=int(len(x_0) * .66), random_state=42)

x = np.vstack((x_0, x_1))

del(x_1)
del(x_0)

y = x[:,-1]
x = x[:,:-1]

import xgboost as xgb

dtrain = xgb.DMatrix(x, label=y)

param = {"max_depth": 6, "eta":.3, "verbosity":2}

bst = xgb.train(param, dtrain)

test_x = []
for doc in test_freqs.keys():
    row = []
    for uid in uids:
        if uid in test_freqs[doc].keys():
            row.append(test_freqs[doc][uid])
        else:
            row.append(0)
    test_x.append(row)

test_x = np.array(test_x)

test_y = []
for doc in test_freqs.keys():
    if "D000818" in solution[doc]:
        test_y.append(1)
    else:
        test_y.append(0)
        
test_y = np.array(test_y)

dtest = xgb.DMatrix(test_x)

ypred = bst.predict(dtest)

from sklearn import metrics
print(metrics.accuracy_score(test_y, np.round(ypred)))

ypred = list(np.round(ypred))

true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0

for idx in range(len(test_y)):
    if predictions[idx] == 1 and test_y[idx] == 1:
        true_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 1:
        false_neg += 1
    if predictions[idx] == 1 and test_y[idx] == 0:
        false_pos += 1
    if predictions[idx] == 0 and test_y[idx] == 0:
        true_neg += 1

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
f1 = (2 * precision * recall) / (precision + recall)

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
"""
    

##################### rf - f1 - testing with subset of terms

import json
import numpy as np
import time

subset = []
# load in subset terms list
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        subset.append(line.strip("\n"))
subset = set(subset)
        
# Load in term frequencies
with open("./data/term_freqs_rev_1.json", "r") as handle:
    temp = json.load(handle)
 
uids = []

with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        if line[0] in subset:
            uids.append(line[0])
        
docs_list = list(temp.keys())
partition = int(len(docs_list) * .8)

train_docs = docs_list[0:partition]
test_docs = docs_list[partition:]

# Load in solution values
solution = {}
docs_list = set(docs_list)
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:            
            solution[line[0]] = [term for term in line[1:] if term in subset]

train_docs = [doc for doc in train_docs if doc in solution.keys()]
test_docs = [doc for doc in test_docs if doc in solution.keys()]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]

terms_to_test = list(subset)

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from notify import notify

for term_to_test in terms_to_test:
    start = time.perf_counter()
    y = []
    for doc in train_docs:
        if term_to_test in solution[doc]:
            y.append(1)
        else:
            y.append(0)
    
    y = np.array(y)
    
    x_0 = []
    x_1 = []
    for doc in train_docs:
        row = []
        for uid in uids:
            if uid in train_freqs[doc].keys():
                row.append(train_freqs[doc][uid])
            else:
                row.append(0)
        row.append(y[train_docs.index(doc)])
        if row[-1] == 0:
            x_0.append(row)
        else:
            x_1.append(row)
    
    x_0 = np.array(x_0)
    x_0 = x_0[0:20000]
    
    x_1 = np.array(x_1)
    
    x_1 = resample(x_1, replace=True, n_samples=int(len(x_0) * .66), random_state=42)
    
    x = np.vstack((x_0, x_1))
    
    del(x_1)
    del(x_0)
    
    y = x[:,-1]
    x = x[:,:-1]

    clf = RandomForestClassifier(verbose=0, random_state=42, n_estimators=100)
    
    clf.fit(x, y)
    print("elapsed: " + str(time.perf_counter() - start))
    test_x = []
    for doc in test_freqs.keys():
        row = []
        for uid in uids:
            if uid in test_freqs[doc].keys():
                row.append(test_freqs[doc][uid])
            else:
                row.append(0)
        test_x.append(row)
    
    test_x = np.array(test_x)
    
    test_y = []
    for doc in test_freqs.keys():
        if term_to_test in solution[doc]:
            test_y.append(1)
        else:
            test_y.append(0)
            
    test_y = np.array(test_y)
    
    predictions = clf.predict(test_x)
    
    clf.score(test_x, test_y)
    
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    
    for idx in range(len(test_y)):
        if predictions[idx] == 1 and test_y[idx] == 1:
            true_pos += 1
        if predictions[idx] == 0 and test_y[idx] == 1:
            false_neg += 1
        if predictions[idx] == 1 and test_y[idx] == 0:
            false_pos += 1
        if predictions[idx] == 0 and test_y[idx] == 0:
            true_neg += 1
    
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    notify("term: " + term_to_test + " f1: " + str(f1))
