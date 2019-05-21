#!/usr/bin/env python3

import json

import pandas as pd

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
        
df = pd.DataFrame(columns=uids)

for doc in term_freqs:
    temp_dict = dict({'0': doc[0]}, **doc[1])
    doc_df = pd.DataFrame([temp_dict], columns=uids)
    df = pd.concat([df, doc_df])

df.to_csv("./data/term_freqs.csv")