#!/usr/bin/env python3

import json

import pandas as pd
from tqdm import tqdm

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

dfs = []

for doc in tqdm(term_freqs[0:30]):
    temp_dict = dict({'0': doc[0]}, **doc[1])
    doc_df = pd.DataFrame([temp_dict], columns=uids)
    dfs.append(doc_df)

df = pd.concat(dfs)
#df.to_csv("./data/term_freqs.csv")

# get a solution for one class
y = []
for doc in term_freqs[0:30]:
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

selected_feats = []
counter = 0
for feat in feat_selector.support_:
    if feat == True:
        selected_feats.append(counter)
    counter += 1

for feat in selected_feats:
    print(uids[feat])
    
feat_selector.ranking_[0:10]

X_filtered = feat_selector.transform(x)
