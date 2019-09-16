import json
import numpy as np
import logging


print("working on input")
subset = []

        
with open("./data/term_freqs.json", "r") as handle:
    temp = json.load(handle)

temp = {i[0]: i[1] for i in temp}

top_500 = []
with open("./data/top_500_terms", "r") as handle:
    for line in handle:
        top_500.append(line.strip("\n"))


docs_list = list(temp.keys())
partition = int(len(docs_list) * .9)

train_docs = docs_list[0:partition]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]


x = []
for doc in train_docs:
    row = []
    for uid in top_500:
        if uid in train_freqs[doc].keys():
            row.append(train_freqs[doc][uid])
        else:
            row.append(0)
    x.append(row)

x_np = np.array(x)

zs = []
for samp in x:
    if max(samp) > 0:
    #samp = [val for val in samp if val > 0]
        mean = sum(samp) / len(samp)
        std_dev = (1/len(samp) * sum([(samp_i - mean)**2 for samp_i in samp]))**0.5
        z_samp = [(x_i - mean) / std_dev for x_i in samp]
        zs.append(z_samp)
minmaxs = []
for samp in zs:
    minmax = [(x_i - min(samp)) / (max(samp) - min(samp)) for x_i in samp]
    minmaxs.append(minmax)

minmaxs = np.array(minmaxs)
#zs = (x - x.mean()) / x.std()

#minmax = (zs - zs.min()) / (zs.max() - zs.min())

# after getting temp in
train_freqs = {}
for sample in temp:
    freqs = temp[sample].items()
    mean = sum(freqs.values()) / 29351
    std_dev = (1/29351 * sum([(x_i - mean)**2 for x_i in freqs.values()]))**0.5
    for freq in freqs:
        freqs[freq] = (freqs[freq] - mean) / std_dev
    min_freq = min(freqs.values())
    max_freq = max(freqs.values())
    for freq in freqs:
        freqs[freq] = (freqs[freq] - min_freq) / (max_freq - min_freq)

